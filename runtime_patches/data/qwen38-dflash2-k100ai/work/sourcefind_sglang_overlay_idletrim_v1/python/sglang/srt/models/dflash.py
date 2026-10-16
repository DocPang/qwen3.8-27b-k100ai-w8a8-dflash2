# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.speculative.dflash2_selector import selector_walk_triton
from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    parse_dflash_draft_config,
)

logger = logging.getLogger(__name__)


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.layer_id = int(layer_id)
        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            prefix="o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = float(
                rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000))
            )
            rope_scaling = dict(rope_parameters)
        else:
            rope_theta = float(getattr(config, "rope_theta", 10000))
            rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        # DFlash2 Qwen3.8 was trained with sliding_attention (HF window=2048)
        # and non-causal block attention. Keep the per-layer RadixAttention
        # metadata exactly aligned with the model-level value consumed by the
        # Triton backend planner.
        layer_types = get_dflash_layer_types(config)
        if layer_types is None:
            sliding_window_size = -1
        else:
            if layer_id >= len(layer_types):
                raise ValueError(
                    "DFLASH config.layer_types must contain one entry per draft layer. "
                    f"Got {len(layer_types)} entries, layer_id={layer_id}."
                )
            layer_type = layer_types[layer_id]
            if layer_type == "full_attention":
                sliding_window_size = -1
            elif layer_type == "sliding_attention":
                resolved_window = get_dflash_attention_sliding_window_size(config)
                if resolved_window is None:
                    raise ValueError(
                        "DFLASH sliding_attention layer requires a sliding window."
                    )
                sliding_window_size = int(resolved_window)
            else:
                raise ValueError(
                    "Unsupported DFLASH draft layer type. "
                    f"layer_types[{layer_id}]={layer_type!r}."
                )
        is_causal = getattr(config, "is_causal", None)
        attn_type = (
            AttentionType.DECODER
            if is_causal is True
            else AttentionType.ENCODER_ONLY
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            attn_type=attn_type,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        debug_dump_dir = os.getenv("SGLANG_DFLASH_LAYER_DUMP_DIR", "").strip()
        debug_l0 = (
            bool(debug_dump_dir)
            and self.layer_id == 0
            and int(hidden_states.shape[0]) == 8
            and int(positions[0].item()) == 30
        )
        if debug_l0:
            os.makedirs(debug_dump_dir, exist_ok=True)
            torch.save(q.detach().cpu(), os.path.join(debug_dump_dir, "layer0_q_pre_norm.pt"))
            torch.save(k.detach().cpu(), os.path.join(debug_dump_dir, "layer0_k_pre_norm.pt"))
            torch.save(v.detach().cpu(), os.path.join(debug_dump_dir, "layer0_v_pre_rope.pt"))
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        if debug_l0:
            torch.save(q.detach().cpu(), os.path.join(debug_dump_dir, "layer0_q_post_norm.pt"))
            torch.save(k.detach().cpu(), os.path.join(debug_dump_dir, "layer0_k_post_norm.pt"))
        q, k = self.rotary_emb(positions, q, k)
        if debug_l0:
            torch.save(q.detach().cpu(), os.path.join(debug_dump_dir, "layer0_q_post_rope.pt"))
            torch.save(k.detach().cpu(), os.path.join(debug_dump_dir, "layer0_k_post_rope.pt"))
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _grouped_conv_eager(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """DFlash2 grouped dynamic depthwise convolution.

    This first K100AI backport intentionally keeps the mathematically equivalent
    eager PyTorch implementation instead of upstream torch.compile so correctness
    can be qualified on gfx928 before graph/compiler tuning.
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"DFLASH conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = hidden_size // self.group_size
        base_kernel = torch.zeros(2, self.taps, hidden_size)
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(self, hidden_states, delta, side: int) -> torch.Tensor:
        return _grouped_conv_eager(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(self, hidden_states: torch.Tensor, coefficients) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        attention_conv: Optional[DFlashGroupedConv] = None,
        mlp_conv: Optional[DFlashGroupedConv] = None,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.layer_id = int(layer_id)

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config)
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)

        debug_dump_dir = os.getenv("SGLANG_DFLASH_LAYER_DUMP_DIR", "").strip()
        debug_this = False
        if debug_dump_dir and int(positions.numel()) == 8:
            debug_this = int(positions[0].item()) == 30
        if debug_this:
            os.makedirs(debug_dump_dir, exist_ok=True)
            torch.save(
                hidden_states.detach().to("cpu"),
                os.path.join(debug_dump_dir, f"layer{self.layer_id}_attn_input.pt"),
            )

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        if debug_this:
            torch.save(
                attn_out.detach().to("cpu"),
                os.path.join(debug_dump_dir, f"layer{self.layer_id}_attn_raw.pt"),
            )
        if os.getenv("SGLANG_DFLASH2_MODEL_STAGE_SYNC", "0").strip().lower() in ("1", "true", "yes", "on"):
            torch.cuda.synchronize()
            logger.info("DFLASH2_MODEL_STAGE layer=%d attention_done", self.self_attn.layer_id)
        if attention_kernel is not None:
            attn_out = self.attention_conv.finish(attn_out, attention_kernel)
        if debug_this:
            torch.save(
                attn_out.detach().to("cpu"),
                os.path.join(debug_dump_dir, f"layer{self.layer_id}_attn_postconv.pt"),
            )

        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if os.getenv("SGLANG_DFLASH2_MODEL_STAGE_SYNC", "0").strip().lower() in ("1", "true", "yes", "on"):
            torch.cuda.synchronize()
            logger.info("DFLASH2_MODEL_STAGE layer=%d mlp_done", self.self_attn.layer_id)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        if debug_this:
            torch.save(
                (hidden_states + residual).detach().to("cpu"),
                os.path.join(debug_dump_dir, f"layer{self.layer_id}_full_output.pt"),
            )
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        draft_config = self.draft_config = parse_dflash_draft_config(
            draft_hf_config=config
        )
        self.block_size = int(draft_config.resolve_block_size(default=16))
        self.candidate_selector = None

        def grouped_conv():
            if not draft_config.conv_kernel_size:
                return None
            return DFlashGroupedConv(
                hidden_size,
                self.block_size,
                draft_config.conv_kernel_size,
                draft_config.conv_group_size,
            )

        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config=config,
                    layer_id=i,
                    attention_conv=grouped_conv(),
                    mlp_conv=grouped_conv(),
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        if draft_config.num_target_layers is not None:
            target_num_layers = int(draft_config.num_target_layers)
        elif draft_config.target_layer_ids is not None:
            target_num_layers = max(draft_config.target_layer_ids) + 1
        else:
            target_num_layers = num_layers
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = int(block_size)
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None:
                    conv.block_size = self.block_size

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Return SGLang window_left for the draft attention backend planner."""
        return get_dflash_attention_sliding_window_size(self.config)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_source_weights = 0
        ignored_source_weights: list[str] = []

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_source_weights += 1
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    ignored_source_weights.append(name)
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_source_weights += 1

        if isinstance(self, DFlash2DraftModel):
            if ignored_source_weights:
                raise RuntimeError(
                    "DFlash2 checkpoint contained weights not consumed by the SourceFind backport: "
                    + ", ".join(ignored_source_weights)
                )
            expected = 15 * len(self.layers) + 6
            if loaded_source_weights != expected:
                raise RuntimeError(
                    "DFlash2 checkpoint load-count mismatch: "
                    f"loaded={loaded_source_weights}, expected={expected}."
                )
            logger.info(
                "DFlash2 checkpoint strict load audit passed: %d/%d source tensors consumed.",
                loaded_source_weights,
                expected,
            )


class CandidateSelector(nn.Module):
    """DFlash2 low-rank candidate transition scorer.

    The first K100AI bring-up intentionally keeps lattice construction eager while
    retaining the upstream Triton path walk. This isolates compiler tuning from
    correctness and preserves the checkpoint's exact parameter names.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        keys = self.successor_codebook[candidate_ids]
        predecessor_ids = torch.cat(
            [
                anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
                candidate_ids[:, :-1],
            ],
            dim=1,
        )
        predecessors = self.predecessor_codebook[predecessor_ids]
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
        )

    def sample_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
    ):
        return selector_walk_triton(
            candidate_ids=candidate_ids,
            scores=scores,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash2 backbone + candidate selector for the SourceFind 0.5.12 backport."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        draft_config = self.draft_config
        if not draft_config.selector_rank:
            raise ValueError(
                "DFlash selector draft requires dflash_config.selector_rank."
            )
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=draft_config.selector_rank,
            top_k=draft_config.selector_top_k,
        )
        self.lm_head: Optional[nn.Module] = None

    def _transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.draft_config.output_multiplier != 1.0:
            logits.mul_(self.draft_config.output_multiplier)
        softcap = self.draft_config.final_logit_softcapping
        if softcap is not None:
            logits.div_(softcap).tanh_().mul_(softcap)
        return logits

    def compute_candidates(self, hidden: torch.Tensor):
        """Compute global DFlash2 selector candidates without gathering full logits.

        SourceFind's target LM head is vocab-parallel under TP.  Each rank computes
        top-k on its local *unpadded* base-vocab shard, converts local ids to global
        token ids, gathers only ``tp_size * k`` values/ids, and applies the final
        global top-k identically on every rank.  TP1 keeps the existing fast path.
        """
        assert self.lm_head is not None, "draft_model.lm_head unset"
        weight = getattr(self.lm_head, "weight", None)
        if weight is None or weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise RuntimeError(
                "DFlash2 selector requires a dense FP16/BF16/FP32 target lm_head."
            )

        hidden = hidden.to(weight.dtype)
        top_k = int(self.candidate_selector.top_k)
        tp_size = int(get_tensor_model_parallel_world_size())
        shard = getattr(self.lm_head, "shard_indices", None)

        if tp_size == 1:
            org_vocab_size = int(
                getattr(self.lm_head, "org_vocab_size", int(self.config.vocab_size))
            )
            if shard is not None:
                org_vocab_size = int(shard.num_org_elements)
            if os.getenv("SGLANG_DFLASH2_CHUNKED_TOPK", "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                chunk = int(os.getenv("SGLANG_DFLASH2_CHUNKED_TOPK_SIZE", "8192"))
                best_vals = None
                best_ids = None
                for start in range(0, org_vocab_size, chunk):
                    end = min(org_vocab_size, start + chunk)
                    local_logits = torch.matmul(hidden, weight[start:end].T)
                    local_k = min(top_k, end - start)
                    vals, ids = torch.topk(local_logits, local_k, dim=-1)
                    ids = ids.long() + start
                    if best_vals is None:
                        best_vals, best_ids = vals, ids
                        continue
                    merged_vals = torch.cat((best_vals, vals), dim=-1)
                    merged_ids = torch.cat((best_ids, ids), dim=-1)
                    best_vals, sel = torch.topk(merged_vals, top_k, dim=-1)
                    best_ids = torch.gather(merged_ids, -1, sel)
                assert best_vals is not None and best_ids is not None
                return best_ids.long(), self._transform_unary_logits(best_vals)

            logits = torch.matmul(hidden, weight[:org_vocab_size].T)
            vals, ids = torch.topk(logits, top_k, dim=-1)
            return ids.long(), self._transform_unary_logits(vals)

        if shard is None:
            raise RuntimeError(
                "DFlash2 TP candidate projection requires lm_head.shard_indices."
            )
        num_org = int(shard.num_org_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        if num_org <= 0:
            raise RuntimeError("DFlash2 TP rank owns no base-vocabulary lm_head rows.")

        local_k = min(top_k, num_org)
        local_logits = torch.matmul(hidden, weight[:num_org].T)
        local_vals, local_ids = torch.topk(local_logits, local_k, dim=-1)
        if local_k != top_k:
            pad = top_k - local_k
            local_vals = F.pad(local_vals, (0, pad), value=float("-inf"))
            local_ids = F.pad(local_ids, (0, pad), value=0)
        local_ids = local_ids.long() + org_vocab_start

        # Flatten before SourceFind's ProcessGroup all_gather_into_tensor.  This is
        # compatible with its existing DFlash TP greedy path and avoids full-vocab
        # communication.  Result order is rank-major, then token, then top-k.
        tp_group = get_tp_group()
        nrows = int(local_vals.shape[0])
        elems_per_rank = nrows * top_k
        gathered_vals = torch.empty(
            tp_size * elems_per_rank,
            dtype=local_vals.dtype,
            device=local_vals.device,
        )
        gathered_ids = torch.empty(
            tp_size * elems_per_rank,
            dtype=local_ids.dtype,
            device=local_ids.device,
        )
        tp_group.all_gather_into_tensor(gathered_vals, local_vals.contiguous().view(-1))
        tp_group.all_gather_into_tensor(gathered_ids, local_ids.contiguous().view(-1))

        gathered_vals = (
            gathered_vals.view(tp_size, nrows, top_k)
            .permute(1, 0, 2)
            .reshape(nrows, tp_size * top_k)
        )
        gathered_ids = (
            gathered_ids.view(tp_size, nrows, top_k)
            .permute(1, 0, 2)
            .reshape(nrows, tp_size * top_k)
        )
        top_vals, sel = torch.topk(gathered_vals, top_k, dim=-1)
        top_ids = torch.gather(gathered_ids, -1, sel)
        return top_ids.long(), self._transform_unary_logits(top_vals)


EntryClass = [DFlashDraftModel, DFlash2DraftModel]
