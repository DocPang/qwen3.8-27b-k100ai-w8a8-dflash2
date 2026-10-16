"""Wire the bit-exact fused GDN in-proj split into the live model (runtime overlay).

Problem (static analysis, qwen3_5.py:462-513)
  if self.num_v_heads // self.num_k_heads in [1, 2, 4]:  -> vendor fused kernel
  else:                                                  -> SLOW branch, every GDN layer:
        b = b.contiguous(); a = a.contiguous()      (2 launches, tensors are only S x 24)
        mixed_qkv = torch.cat((query,key,value),-1) (1 launch)
  Our ratio is 48/16 = 3, so all 48 GDN layers take the slow branch: ~144 launches/step
  (device census: direct_copy 228 calls/1.425 ms, cat_copy 48.5 calls/0.345 ms).

  The vendor kernel is NOT reusable at ratio 3: its docstring states it reads the
  per-head-group INTERLEAVED layout, while this checkpoint stores contiguous blocks
  [q|k|v|z] -- the [1,2,4] gate is a correctness guard, not an oversight.

Fix (this overlay)
  Re-implement the slow branch as ONE triton launch per layer via
  $D/gdn_split_fused.py, validated bit-exact for S in {1,8,16,32} against a
  byte-for-byte replica of the reference branch.  Pure data movement => no numeric risk.
  forward() is wrapped only for the exact guarded case (ratio 3, bf16, cuda); the wrapper
  reuses self._forward_input_proj and replicates the original 12-line tail verbatim, and
  z is returned as a (S, nv_tp, head_v) view so z_shape_og is identical to the original.

Fail-closed: inert without gdn_fuse_cfg.txt; inert if the pinned qwen3_5.py md5 differs;
guards return the parent forward otherwise; one env/gate switch reverts everything.
"""
import importlib.util
import os
import runpy

_EXP = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_ON = f"{_EXP}/gdn_fuse_cfg.txt"
_PARENT = f"{_EXP}/tail_overlay/sitecustomize.py"
_KERNEL = f"{_EXP}/gdn_split_fused.py"
_PIN_MD5 = "c3dfd2a282c5df010a5be2532932765e"   # sglang/srt/models/qwen3_5.py
_TARGET_FILE = ("/usr/local/lib/python3.10/dist-packages/sglang/srt/models/"
                "qwen3_5.py")
_HELPERS = ("run_accuracy", "evalscope", "lm_eval", "pytest", "warm_request.py",
            "probe_c1_m8.py", "gdn_probe.py", "shortqa_probe.py", "accuracy_ref_run.py",
            "gdn_split_fused.py")


def _say(m):
    print("[Q38 GDNFUSE] " + m, flush=True)


if not os.path.exists(_ON):
    _say("inert (no %s)" % _ON)
elif any(h in " ".join(__import__("sys").argv) for h in _HELPERS):
    _say("helper process, inert")
else:
    import hashlib
    import sys

    if os.path.exists(_PARENT):
        try:
            runpy.run_path(_PARENT, run_name="__q38_gdnfuse_parent__")
        except Exception as exc:
            _say(f"baseline chain failed: {exc}")
    try:
        h = hashlib.md5(open(_TARGET_FILE, "rb").read()).hexdigest()
        if h != _PIN_MD5:
            _say("PIN MISMATCH (%s != %s) -> inert" % (h, _PIN_MD5))
        elif not os.path.exists(_KERNEL):
            _say("kernel file missing -> inert")
        else:
            spec = importlib.util.spec_from_file_location("q38_gdn_split_fused", _KERNEL)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

            _prev_forward = Qwen3_5GatedDeltaNet.forward
            _hits = [0]

            def _forward(self, hidden_states, forward_batch):
                try:
                    nk, nv = self.num_k_heads, self.num_v_heads
                    tp = self.attn_tp_size
                    if (
                        nv // nk == 3
                        and tp > 0
                        and hidden_states.dtype is torch.bfloat16
                        and hidden_states.is_cuda
                        and self.head_k_dim == 128
                        and self.head_v_dim == 128
                    ):
                        k_tp = self.key_dim // tp
                        v_tp = self.value_dim // tp
                        nv_tp = nv // tp
                        qkvz, ba = self._forward_input_proj(hidden_states)
                        if tuple(qkvz.shape) == (hidden_states.shape[0], 2 * k_tp + 2 * v_tp) \
                                and tuple(ba.shape) == (hidden_states.shape[0], 2 * nv_tp):
                            mixed_qkv, z, b, a = mod.gdn_split_fused(
                                qkvz, ba, k_tp, v_tp, nv_tp)
                            z = z.view(hidden_states.shape[0], nv_tp, self.head_v_dim)
                            _hits[0] += 1
                            if _hits[0] == 1:
                                _say("fused path HIT: S=%d k_tp=%d v_tp=%d nv_tp=%d"
                                     % (hidden_states.shape[0], k_tp, v_tp, nv_tp))
                            # ---- original tail, replicated verbatim ----
                            core_attn_out = self.attn(forward_batch, mixed_qkv=mixed_qkv,
                                                      a=a, b=b)
                            z_shape_og = z.shape
                            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
                            z = z.reshape(-1, z.shape[-1])
                            if core_attn_out.shape != z.shape:
                                core_attn_out_pad = torch.zeros_like(z)
                                core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
                                core_attn_out = core_attn_out_pad
                            core_attn_out = self.norm(core_attn_out, z)
                            core_attn_out = core_attn_out.reshape(z_shape_og)
                            core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
                            output, _ = self.out_proj(core_attn_out)
                            return output
                except Exception as exc:
                    if _hits[0] == 0:
                        _say("fused path guard failed (%r) -> parent forward" % exc)
                return _prev_forward(self, hidden_states, forward_batch)

            import torch  # noqa: E402  (guards reference torch)

            Qwen3_5GatedDeltaNet.forward = _forward
            _say("installed: bit-exact fused GDN split (pin md5 %s ok)" % h[:12])
    except Exception as exc:
        _say(f"install FAILED (server keeps vendor path): {exc}")
