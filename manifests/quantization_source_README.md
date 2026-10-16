---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- vllm
- qwen3_5
- qwen3.8
- compressed-tensors
- w8a8
- int8
- smoothquant
- quantized
- mtp
- speculative-decoding
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B — SmoothQuant W8A8-INT8 (full-coverage) + BF16 MTP

**The lowest-divergence W8A8-INT8 of this architecture we are aware of: mean
full-vocab KLD vs BF16 = 0.0141** — ~2.8× lower than standard dynamic-INT8
W8A8 recipes (which cluster at 0.039–0.040 on the 3.6 sibling), while keeping
the **native INT8 tensor-core path** (`CutlassInt8ScaledMMLinearKernel`) that
gives Ampere-class GPUs ~+70% prefill over weight-only (Marlin) quants.

Built on release day from `Qwen/Qwen3.8-27B` BF16. Numerics only, no
fine-tuning. All credit for the model to Qwen.

## Why this exists

On SM86 (RTX 3090 class) there are exactly two viable 8-bit families:

| family | kernel path | prefill | typical KLD vs BF16 |
|---|---|---|---|
| W8A16 (weight-only) | Marlin dequant | ~2000 tok/s (2×3090 TP2) | 0.002–0.009 |
| W8A8 dynamic INT8 | **CUTLASS native INT8** | **~3400 tok/s** | **0.039–0.040** |

The ~0.04 W8A8 divergence is almost entirely the **dynamic per-token INT8
activation cost** — we measured weight-observer and layer-exclusion choices to
be worth ~zero (two independent public 3.6 W8A8 quants land within 0.0005 of
each other). The one mechanism that attacks the activation term is
**SmoothQuant** (fold activation outlier channels into the adjacent weights
pre-quantization). This checkpoint applies it to **all four GEMM input
classes**, not just the FFN:

| fold | coverage | mechanism (all mathematically exact pre-quantization) |
|---|---|---|
| A ffn-in | 64 layers | `post_attention_layernorm` → gate/up columns |
| B attn/gdn-in | 64 layers | `input_layernorm` → q/k/v or `in_proj_qkv/z` columns (+ BF16 `in_proj_a/b` compensation) |
| C down-in | 64 layers | up_proj rows ÷ s → down_proj columns (SiLU product is elementwise) |
| D gdn-out | 48 layers | `linear_attn.norm` (gated, per-head-dim) → out_proj columns |
| E attn-out | 16 layers | v_proj rows ÷ t → o_proj columns, GQA-constrained (shared scale per KV group) |

Measured ablation on the 3.6 sibling: FFN-only smoothing = 0.0394 → 0.0351;
full coverage on 3.8 = **0.0141**.

Two implementation details that matter if you reproduce this (both cost us a
broken build to learn):

1. **`Qwen3_5RMSNorm` is zero-centered** (`out = x̂ · (1 + w)`). The textbook
   SmoothQuant fold `w/s` silently produces garbage; the correct fold is
   `w' = (1+w)/s − 1`. (`Qwen3_5RMSNormGated` in the GDN blocks is plain
   `x̂ · w`, where `w/s` IS correct.)
2. Storing the zero-centered fold in bf16 destroys precision near gain≈0
   (up to 14% per-channel error at high s). The smoothed intermediate is
   therefore stored **float16** with **s capped at 16**.

## Recipe

- **Smoothing**: custom file-to-file transform (scripts in `quantization/`),
  α = 0.8, per-channel abs-max over 96×2048-token WikiText-2 windows,
  s ∈ [1e-5, 16]. Verified per-fold against runtime semantics in fp64
  (max functional deviation ≈ 1e-4, i.e. fp16 rounding).
- **Quantization**: [llm-compressor](https://github.com/vllm-project/llm-compressor)
  `QuantizationModifier`, scheme **W8A8** — INT8 weights per-channel symmetric
  (minmax), **INT8 dynamic per-token activations** → `compressed-tensors`
  `int-quantized`.
- **Preserved BF16**: `mtp.*` (the multi-token-prediction drafter — quantizing
  it silently destroys speculative decoding), `lm_head`, `embed_tokens`,
  vision tower, GDN `in_proj_a`/`in_proj_b` recurrent gates, all norms.
- MTP head grafted back post-export (`model-mtp.safetensors`, 15 tensors) —
  transformers' save path drops modules the modeling class does not register,
  and its ignore-list serialization drops the `mtp` pattern; the shipped
  `config.json` carries the corrected ignore list.

## Measured quality (full-vocab KLD, 122,640 positions, 240×512 WikiText-2 windows, top-K 512, vs BF16 teacher-forced reference)

| checkpoint (each vs its own BF16) | mean KLD | median | top-1 agree | P99.9 / max | PPL ln ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 public Quark W8A8 | 0.03994 | 0.0152 | 92.37% | 1.95 / 24.7 | +0.0208 |
| Qwen3.6 public llm-compressor W8A8 | 0.03949 | — | 92.37% | 2.01 / 26.1 | — |
| Qwen3.6 SmoothQuant FFN-only W8A8 (ablation) | 0.03508 | 0.0125 | 92.95% | 1.88 / 19.8 | +0.0175 |
| **this checkpoint (full-coverage SQ)** | **0.01414** | **0.0058** | **95.32%** | **0.44 / 17.2** | **+0.0069** |
| Qwen3.6 AutoRound W8A16 (reference class) | 0.00244 | 0.0009 | 97.89% | 0.087 / 5.0 | −0.0016 |

The tail is the headline: P99.9 per-position KLD is **4.5× lower** than
standard W8A8 — the catastrophic-divergence positions are mostly gone.

## Verified (vLLM, TP2 on 2×RTX 3090, 330 W caps)

| check | result |
|---|---|
| kernel | `CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8` |
| KV pool | 373,445 tokens @ 7.13 GiB/GPU pin, fp8_e4m3 KV, 262,144 max len |
| prefill 2K/8K/16K | 3425 / 3510 / 3371 tok/s |
| decode (N=1) | 68 tok/s narrative / 108 tok/s code |
| aggregate (6 concurrent) | 330–450 tok/s, 0 preemptions |
| MTP spec decode (K=3) | acceptance length 2.79 — BF16 drafter fully functional |
| tool calls (`qwen3_coder` parser) | clean incl. commas-inside-strings + nested arrays of objects |
| `reasoning_effort` | monotone (xhigh > medium > low), `enable_thinking=false` safe |

⚠ `num_speculative_tokens` **> 3 is not recommended** on hybrid-GDN models:
K=5 reproduced the known illegal-memory-access crash class under concurrent
load on 2×3090 (vLLM #37035 family) in our testing.

## Serve

```bash
vllm serve <this-repo> \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --quantization compressed-tensors \
  --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Sampling per the upstream card: temperature 1.0, top_p 0.95, top_k 20.
Reasoning effort via `reasoning_effort: xhigh|medium|low` (other values are
rejected by the chat template).

## Files

- `model-0000N-of-00015.safetensors` — W8A8-INT8 weights + BF16 exclusions
- `model-mtp.safetensors` — BF16 MTP drafter (15 tensors)
- `config.json` — includes the corrected quantization ignore list (do not
  regenerate it from the recipe; see notes above)
- `recipe.yaml` — llm-compressor stage (quantization step only)
- `quantization/` — the full smoothing pipeline: activation capture,
  fold transform (+ fp64 verifier), build and MTP-graft scripts
- `chat_template.jinja`, tokenizer files — from upstream

## License

Apache-2.0 for this packaging; the base model is Apache-2.0 by Qwen.
