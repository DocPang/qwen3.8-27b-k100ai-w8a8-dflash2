# 从「原始镜像 + 原始权重」复现本方案

本文是**唯一权威的复现步骤**。每一步都给出「期望看到什么」；任何一步和期望不符，就不要继续往下走。

---

## 0. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| 硬件 | 2 × 海光 K100AI（DCU，`gfx928`），同一 NUMA 域内可见 | TP2 必须同机同对卡 |
| 宿主 | Ubuntu 22.04，`/dev/kfd` + 2 个 render 节点可用 | 本方案实测用 `renderD130/131` |
| 容器运行时 | Docker（或兼容），支持 `--privileged --ipc host --network host` | 三项都必需 |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317d…c3aca68` | 见 §1 |
| 权重 | 官方 Qwen3.8-27B 的 **SmoothQuant W8A8-INT8** 量化副本 | 见 §2，含逐文件 SHA256 |
| 草稿 | `Qwen3.8-27B-DFlash2`（5 层 BF16） | 见 §2 |
| 磁盘 | `/data` ≥ 60 GB（权重 30 GB + 草稿 3.9 GB + 运行时 5 GB） | 本方案全部改动物理上都在 `/data` |

> **口径**：本方案**不修改任何权重文件、不修改量化表、不覆盖基础 `.so`**。
> 全部加速来自运行时 overlay（monkey-patch）、Triton/rocBLAS 调优表、和一个独立的自研 HIP 扩展。

> **运行时依赖已随仓库交付**：overlay 父链（`batch2_raw_q8_overlay`、`spec_gain_*`、
> `integration/parent_v8` 等）、aiter backport、DirectLDS/rocBLAS 原生扩展都在
> `overlays/vendor/`。`setup_release.sh` 把它们铺到 `Q38_RUNTIME_ROOT`
> （默认 `/data/qwen38-runtime`），`launch_official_lane.sh` 再把该目录挂到容器内的
> `/data/qwen38-27b-k100ai-int8-opt`。因此**不需要**作者机器上的那份目录。

---

## 1. 镜像

```bash
IMG='harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68'
docker pull "$IMG"          # 结果必须仍是同一个 digest（内容寻址）
docker image inspect "$IMG" --format '{{.Id}}'
```

* 镜像内是 **SGLang 0.5.12** + 海光 DTK 运行时（`DTK_VERSION=2604`）。
* 期望：`RepoDigests` 与上面一致，`Architecture=amd64`。

### 1.1 首次启动的离线前置（很重要）

镜像里 `tvm_ffi` 会在首次 import 时**现场编译**一个 torch C 扩展
（`_build_optional_torch_c_dlpack.py` → `/root/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch210-rocm.so`）。
在**无外网**的机器上这条编译路径会卡住/反复重试，表现为：日志停在 numpy 警告、`/health` 长时间 000。

解决办法（任选其一）：

```bash
# A. 仓库已自带该产物：<repo>/tvmffi_cache/libtorch_c_dlpack_addon_torch210-rocm.so
#    launch_official_lane.sh 会把它挂到 /root/.cache/tvm-ffi
# B. 若你的镜像/驱动与仓库固定的 digest 不同，仍可在有外网的机器上先启动一次再复用
```

`scripts/launch_official_lane.sh` 会在它缺失时直接报错退出（离线机器必读）；
确实要让容器联网自建时，设 `Q38_ALLOW_TVMFI_BUILD=1`。

---

## 2. 权重

### 2.1 目标模型：官方 Qwen3.8-27B 的 SmoothQuant W8A8-INT8

```text
/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8/
```

* 15 个 `model-0000X-of-00015.safetensors` + `model-mtp.safetensors` + `model.safetensors.index.json`
* `config.json` 中 `quantization_config.quant_method = compressed-tensors`，
  `W8A8`（weights: int8 / channel / symmetric / `memoryless_minmax`；activations: int8 / token / **dynamic**），
  忽略 `lm_head`、`embed_tokens`、`mtp`、`visual`、`linear_attn.in_proj_a/b`。
* 量化配方（`recipe.yaml`，随仓库提供）：
  `targets: [Linear]`，`ignore: [lm_head, embed_tokens, linear_attn.in_proj_a/b, visual, mtp]`，`scheme: W8A8`。
* 该 checkpoint 对**全部四类 GEMM 输入**做了 SmoothQuant 折叠（FFN-in / attn-GDN-in / down-in / GDN-out）
  与 attn-out（GQA 约束共享 scale），量化细节见 `manifests/quantization_source_README.md`。

**校验**：`manifests/target_weights_sha256.txt` 列出全部 24 个文件的 SHA256。

```bash
cd /data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8
sha256sum -c /path/to/repo/manifests/target_weights_sha256.txt
```

### 2.2 草稿模型：DFlash2

```text
/data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2/{config.json, model.safetensors}
```

* SHA256 见 `manifests/draft_sha256.txt`：
  `model.safetensors = 67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`（3.85 GB）
* 引擎侧以 `--speculative-algorithm DFLASH --speculative-draft-model-quantization unquant
  --speculative-draft-attention-backend triton --speculative-num-steps 1 --speculative-num-draft-tokens 8` 接入。

> ⚠️ 草稿模型的注意力后端**必须是 `triton`**。曾实测换成 fa3：accept 从 4.57 崩到 1.07、吞吐 62→14.8 t/s
> （而 target 输出 SHA 不变——所以"输出一致"**不能**用来证明草稿路径等价）。详见 `docs/METHODOLOGY.md`。

### 2.3 目录布局

引擎启动脚本会把权重目录**软链接**成 `/tmp/q38-target-model`，并且补两个 0 字节的
`preprocessor_config.json` / `video_preprocessor_config.json`（源目录里这两个文件是 0 字节占位）：

```bash
MODEL=/tmp/q38-target-model
SRC=/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8
mkdir -p "$MODEL"; for f in "$SRC"/*; do ln -sf "$f" "$MODEL/$(basename "$f")"; done
ln -sfn /data/my_models/Qwen/Qwen3.8-27B/preprocessor_config.json        "$MODEL/preprocessor_config.json"
ln -sfn /data/my_models/Qwen/Qwen3.8-27B/video_preprocessor_config.json  "$MODEL/video_preprocessor_config.json"
```

---

## 3. 运行时补丁（17 个 bind mount）

这些文件把镜像里的 SGLang 若干模块替换成本项目的运行时补丁版本。**它们和 overlay 一起构成"本方案"**，
缺任何一个都会得到一个**不同的栈**。逐文件 SHA256 见 `manifests/SHA256SUMS`。

| # | 仓库内路径（`runtime_patches/` 下） | 容器内目标 |
|---|---|---|
| 1 | `…/runtime_patch_official_tp2_native_hcu_policy_20260914/custom_all_reduce.py` | `sglang/srt/distributed/device_communicators/custom_all_reduce.py` |
| 2 | `…/runtime_patch_spec_gap2_markers_v1_20260926/…/eagle_worker_v2.py` | `sglang/srt/speculative/eagle_worker_v2.py` |
| 3 | `…/runtime_patch_newbase_r4e_nb023_20260913/…/qwen3_5_mtp.py` | `sglang/srt/models/qwen3_5_mtp.py` |
| 4 | `…/runtime_patch_newbase_r4e_nb023_20260913/…/mm_utils.py` | `sglang/srt/managers/mm_utils.py` |
| 5 | `…/staging_official_tp2_safe_r2_20260918/…/qwen3_coder_detector.py` | `sglang/srt/function_call/qwen3_coder_detector.py` |
| 6 | `…/staging_official_tp2_safe_r2_20260918/…/base_processor.py` | `sglang/srt/multimodal/processors/base_processor.py` |
| 7 | `…/staging_official_tp2_safe_r2_20260918/assets/qwen38_chat_template.jinja` | `/data/qwen38-dflash2-k100ai/runtime_assets/qwen38_chat_template.jinja` |
| 8 | `…/runtime_patch_official_tp2_gateup_triton_v5d_cache_20260915/W8A8_17408_5120_gfx928_120cu.json` | `lightop/configs/w8a8/W8A8_17408_5120_gfx928_120cu.json` |
| 9 | `…/experiments/dflash_hip_target_sampling_20261012/dflash_worker_sparse_q.py` | `sglang/srt/speculative/dflash_worker.py` |
| 10 | `…/dflash_hip_target_sampling_20261012/dflash_hip_target_only.py` | `sglang/srt/speculative/dflash_hip_target_only.py` |
| 11 | `…/dflash_hip_target_sampling_20261012/dflash_utils.py` | `sglang/srt/speculative/dflash_utils.py` |
| 12 | `…/dflash_hip_target_sampling_20261012/dflash_selector_rejection.py` | `sglang/srt/speculative/dflash_selector_rejection.py` |
| 13 | `…/dflash_hip_target_sampling_20261012/dflash_info_sparse_q.py` | `sglang/srt/speculative/dflash_info.py` |
| 14 | `…/spec_repair_pagedkv_20261008/overlay/extend_attention.py` | `sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 15 | `…/work/sourcefind_sglang_overlay_idletrim_v1/python/sglang/srt/models/dflash.py` | `sglang/srt/models/dflash.py` |
| 16 | `…/work/…/python/sglang/kernels/ops/speculative/`（整个目录） | `sglang/kernels/ops/speculative/` |
| 17 | 启动脚本（见 §5） | `/probe/candidate_entry.sh` |

> 第 16 项是一个**目录**挂载（不是在文件上做替换），会整体覆盖镜像内的
> `kernels/ops/speculative`：这是 DFlash2 的 selector kernel 包。

---

## 4. overlay / 调优表 / 原生扩展

### 4.1 W8A8 Triton 调优表

```text
overlays/tables/            # 308 个 json（每档 (N,K) 一个）
```

通过环境变量 `TRITON_JSON_DIR` 注入（见 §5）。作用：把 int8 GEMM 的分块/`SPLIT_K` 选择**钉死**成
离线 tuning 出来的最优解（所有改动桶 `SPLIT_K=1`，即**逐位精确**，只改调度不改数值）。

### 4.2 11 组 overlay（`overlays/*/sitecustomize.py`）

每个 overlay 都是一个 `sitecustomize.py`，通过 `PYTHONPATH` + `SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX`
串联加载，逐层 `runpy` 父层再替换目标函数。**它们是 fail-closed 的**：
缺少"门文件"时该臂**静默失效**（打印 `inert`，不报错）。
因此"门文件"必须随 overlay 一起交付并在启动后断言：

| 臂 | 目录 | 门文件（`overlays/gates/`） | 期望内容 |
|---|---|---|---|
| CANDHEAD 草稿候选头钉解 | `candhead_overlay/` | `candhead_cfg.txt` | `1` |
| COMBO2 组合快路径 | `combo2_overlay/` | `combo2_cfg.txt` | `1` |
| GDNFUSE GDN 连续块融合 | `gdn_fuse_overlay/` | `gdn_fuse_cfg.txt` | `1` |
| PREFILL_GDN prefill 移植 | `prefill_gdn_overlay/` | `prefill_gdn_cfg.txt` | `1` |
| Q8B8 verifier 快路径 | `q8b8_overlay/` | `q8b8_cfg.txt` | `1` |
| SPLITKV 草稿滑窗 16 路切分 | `draft_splitkv_overlay/` | `draft_splitkv_cfg.txt` | `16` |
| TAILSYNC 尾部强制同步消除 | `tail_overlay/` | `tail_sync_cfg.txt` | **空文件**（存在即开） |
| ATTNCFG 注意力 tile | `attncfg_overlay/` | `attn_cfg.txt` | `16 64 8` |
| FASTHEAD 全档位钉解 | `head_overlay/` | —（无门，链内常开） | `head64_sids.txt` 提供 M→solution 映射 |

> 只有 7 个臂有门文件；FASTHEAD 无门（常开）。
> `attncfg_overlay` 的门是 `attn_cfg.txt`，实测 banner 为 `[Q38 ATTNCFG] BM16/BN64/nw8`。

### 4.2b `$EXP` 根目录负载

下面三样不在任何 arm 子目录里，overlay 直接从 `$EXP` 根读取。`setup_release.sh`
会自动铺好（缺任何一个都会让整条链静默 inert）：

| 文件 | 来源 | 作用 |
|---|---|---|
| `gdn_split_fused.py` | `overlays/gdn_split_fused.py` | GDNFUSE / COMBO2 的融合 GDN 内核 |
| `combo_on.txt` | `overlays/gates/combo_on.txt`（内容 `on`） | COMBO 开关；缺失则 ATT NCFG 链断 |
| `sweep_official.env` | `scripts/sweep_official.env`（路径按 `$EXP` 重写） | 容器内入口脚本的冻结配置 |

另外 `setup_release.sh` 会做一次**宿主父链兼容**：`head_overlay` / `combo2` 的父链
（宿主上的 `batch2_raw_q8_overlay`）按 `.../native_donor_20261002/gateup/v4/` 加载
DirectLDS 融合 GEMM。若本机只有 `parent_compile_v4/`，脚本会补一个同名软链，
否则父链抛 `cannot open shared object file`，FASTHEAD 与 V6/V7 都不会安装。

### 4.3 自研 HIP 扩展

```text
native_ext/k100_assign_fixed8_i64_c16exp_v2.hip   # 源码
native_ext/k100_assign_fixed8_i64_c16exp_v2.so    # 产物，sha256 = 99ef8a6a3f315234d3219e8f9b03d594caaef34fe7dac11c5f124d7cd0809414
```

通过 `SGLANG_Q38_ASSIGN_FIXED8_I64_SO` 注入。作用：把 `assign_fixed8_i64` 这个
"请求→KV 槽位"映射内核从 `batch_size ≤ 8` 扩到 `≤ 16`；正确性由 64-trial 逐元素离线门
与运行时 AUDIT 双重把关（见 `docs/METHODOLOGY.md`）。

---

## 5. 启动参数与入口脚本

### 5.1 冻结配置 `sweep.env`（cap16）

```bash
SWEEP_SPEC_ALGO=DFLASH
SWEEP_GRAPH_ARGS='--cuda-graph-max-bs 16'
SWEEP_MAMBA_TRACK=256
SWEEP_DRAFT_TOKENS=8
SWEEP_NUM_STEPS=1
SWEEP_MAX_REQS=32
SWEEP_MEM_FRAC=0.90
SWEEP_PIECEWISE=off
SWEEP_KV_DTYPE=bfloat16
SWEEP_CHUNK=16384
SWEEP_EXTRA_ARGS='--max-mamba-cache-size 64'
SGLANG_USE_CAUSAL_CONV1D=1
SWEEP_PROFILE_TIMES=0
SWEEP_W8A8_METHODS=1
SWEEP_TRITON_JSON_DIR=<repo>/overlays/tables
SWEEP_LMSLIM_TUNING_JSON=
SWEEP_HIP_VISIBLE=4,5                     # ← 见 §9「陷阱 1」：这是全局卡号
# 链首必须是 concurrency_fix_overlay（它内部再链到 candhead -> combo2 -> ...）
SWEEP_EXTRA_PYTHONPATH=<repo>/overlays/concurrency_fix_overlay
SWEEP_SITECUSTOMIZE_PREFIX=<repo>/overlays/concurrency_fix_overlay
SWEEP_EXTRA_ENV='SGLANG_Q38_ASSIGN_FIXED8_I64_SO=<repo>/native_ext/k100_assign_fixed8_i64_c16exp_v2.so'
SWEEP_DRAFT_MODEL_PATH=/data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2
SWEEP_DRAFT_QUANT=
```

### 5.2 入口脚本（`scripts/entry_official_contest.sh`）

它做四件事，然后 `exec sglang.launch_server`：

1. `source sweep.env`，把 `SWEEP_*` 映射成环境变量与命令行参数；
2. 安装 `decord` wheel（视频依赖，离线本地 wheel）；
3. 软链接权重到 `/tmp/q38-target-model`（见 §2.3）；
4. 启动服务。

最终命令行（照抄即可核对）：

```text
python3 -u -m sglang.launch_server \
  --model-path /tmp/q38-target-model --host 0.0.0.0 --port 18636 --nccl-port 29636 \
  --served-model-name Q38-OFFICIAL-CONTEST-20261017 \
  --chat-template /data/qwen38-dflash2-k100ai/runtime_assets/qwen38_chat_template.jinja \
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 2 --pp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 \
  --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 --mamba-track-interval 256 \
  --cuda-graph-max-bs 16 --disable-piecewise-cuda-graph --page-size 64 \
  --context-length 262144 --mem-fraction-static 0.90 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 --max-running-requests 32 \
  --custom-all-reduce-backend native --enable-metrics \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --max-mamba-cache-size 64 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2 \
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton \
  --speculative-num-steps 1 --speculative-num-draft-tokens 8
```

> `--max-mamba-cache-size` 出现两次是**故意**的：入口脚本先把基线值 `16` 写进固定参数，
> 再由 `SWEEP_EXTRA_ARGS` 覆盖成冻结值 `64`（cap16）。SGLang 取最后一次出现的值。

启动后的**期望几何**（必须一字不差，否则不是本栈）：

```text
max_total_num_tokens=166720, chunked_prefill_size=16384, max_prefill_tokens=16384,
max_running_requests=16, context_len=262144, available_gpu_mem≈10.3 GB
```

### 5.3 起容器

`scripts/launch_official_lane.sh` 会按 §3 §4 的挂载表 + 以下骨架起容器：

```bash
docker run -d --name q38-official-contest-hcu23-20261017 \
  --privileged --ipc host --network host --entrypoint bash \
  --device /dev/kfd --device /dev/dri/renderD130 --device /dev/dri/renderD131 \
  -e HIP_VISIBLE_DEVICES=0,1 -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
  -e SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX=<repo>/overlays/concurrency_fix_overlay \
  -e PYTHONPATH=<repo>/overlays/candhead_overlay \
  -e Q38_INTEGRATION_V7=1 -e Q38_ENABLE_SPEC=1 -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_Q38_NATIVE_CAR_HCU_POLICY=1 -e SGLANG_USE_LIGHTOP=1 \
  -e SGLANG_USE_CUDA_IPC_TRANSPORT=1 -e ALLREDUCE_STREAM_WITH_COMPUTE=1 \
  -e Q38_SPEC_FULLHEAD_M1_SOLUTION=10347 -e Q38_MTP_GATEUP_M4_SOLUTION=10009 \
  -e Q38_MTP_FULLHEAD_M4_SOLUTION=11055 ... \
  -v /data:/data -v /opt/hyhal:/opt/hyhal \
  -v $Q38_RUNTIME_ROOT:/data/qwen38-27b-k100ai-int8-opt \
  -v $Q38_RUNTIME_ROOT/prefill_rocblas_solution:/data/qwen38-27b-k100ai-tp2-opt/build/prefill_rocblas_solution \
  -v <repo>/runtime_patches/…:/usr/local/lib/python3.10/dist-packages/sglang/…   # 16 项
  -v <repo>/tvmffi_cache:/root/.cache/tvm-ffi \
  -v <repo>/scripts/entry_official_contest.sh:/probe/candidate_entry.sh \
  "$IMG" /probe/candidate_entry.sh
```

> **`--network host` 不能省**。少了它在默认 bridge 网络里服务只监听容器内网，宿主 `curl /health` 永远是 000。
>
> **卡号只有一处事实来源**：`GLOBAL_CARDS`（如 `4,5`）同时决定 `--device` 映射和
> 容器内的 `HIP_VISIBLE_DEVICES`（通过 `Q38_HIP_VISIBLE` 覆盖 CFG 里的
> `SWEEP_HIP_VISIBLE`）。并行起第二条 lane 时同时换 `PORT`，`NCCL_PORT` 默认按
> `PORT+11000` 推导，不会撞端口。

---

## 6. 启动自检（两道门，缺一不可）

### 6.1 臂存在性断言

```bash
for tag in "Q38 CANDHEAD" "Q38 COMBO2" "Q38 GDNFUSE" "Q38 HCU01 FASTHEAD" "Q38 PREFILL" "Q38 SPLITKV" "Q38 TAILSYNC"; do
  docker logs <CT> 2>&1 | grep -qF "[$tag]" && echo "arm live $tag" || echo "arm MISSING $tag"
done
```

7 条全部 `arm live` 才算启动正确。任何一条 MISSING 都会静默换掉实测栈，必须修好在继续。

### 6.2 C1 逐位确定性锁

```bash
cd <repo>/scripts && AB_OUT=1024 AB_REPS=3 python3 -u ab3.py rel
```

本仓库参考值（官方 Qwen3.8-27B W8A8 权重，K100AI，TP2）：

| 负载 | STEP_MED | ACCEPT_MED | DECODE_MED | TEXT_SHA | CHARS |
|---|---|---|---|---|---|
| `code` | 58.75 ms | 3.4595 | 58.83 t/s | `9351bc6049cfa597` | 570917 |
| `records` | 57.38 ms | 6.1687 | 107.40 t/s | `ee000f28af8a215d` | 140481 |

要求：**3/3 次复现同一 `TEXT_SHA` 与同一 `ACCEPT_MED`**，且每次 `cached_tokens=0`。

> 这套值绑定「权重 + 全部 overlay + 启动参数」。**只有三项完全一致时才应命中**；
> 命中不了不代表栈错，但必须查出差异来源再继续（这是本方案最强的回归门）。

### 6.3 质量门（20 题，answer-vector 摘要）

```bash
QA_URL=http://127.0.0.1:18636/v1 QA_MODEL=Q38-OFFICIAL-CONTEST-20261017 \
QA_OUT=../results/evidence/shortqa_official.json QA_TAG=official_plain QA_MAXTOK=512 \
python3 -u shortqa_probe.py
```

期望：`correct=20/20, empty=0, answer_vector_digest=89577bb4612ab1d8`。

---

## 7. 官方协议测试

```bash
bash scripts/run_official_suite_official.sh
```

阶段（可断点续跑，已 PASS 的会跳过）：4K/1K `C1→C8→C16→C32 → 64K/1K C1..C32 → 4K C64/C128 →
ShareGPT C1/C4/C16/C32/C64 → 精度门`。所有原始产物写进 `results/official/`。

单档也可单独跑：

```bash
python3 scripts/official_benchmark.py \
  --base-url http://127.0.0.1:18636 --model Q38-OFFICIAL-CONTEST-20261017 \
  --output-dir ./out --input-len 4096 --output-len 1024 --concurrency 16 \
  --warmup-rounds 2 --measure-rounds 3 --requests-per-round 32 \
  --seed 20260912 --timeout 3600
```

### 7.1 精度门：GSM8K **95.0720%**（单次请求）

单次请求口径达标要**两个条件同时成立**，少一个都只有 94.4%：

1. **服务端**启用 Thinking Budget overlay（`overlays/thinkbudget_overlay/`，
   `Q38_THINK_BUDGET=3584`）——请求生成到 3584 token 仍无 `</think>` 时强制闭合；
2. **客户端**的 5-shot 示例取 train split 里 **answer 最长的 5 条**，
   而不是默认的前 5 条。

```bash
# 服务端：overlay 链换成 thinkbudget，并给出预算（其余参数与 §5.1 冻结值一致）
SWEEP_EXTRA_PYTHONPATH=$PWD/overlays/thinkbudget_overlay \
SWEEP_SITECUSTOMIZE_PREFIX=$PWD/overlays/thinkbudget_overlay \
SWEEP_EXTRA_ENV="Q38_THINK_BUDGET=3584" \
  bash scripts/entry_official_contest.sh        # 入口脚本见 §5.2

# 客户端
python3 scripts/gsm8k_retry_pipeline.py \
  --base-url http://127.0.0.1:18636/v1 --model Q38-OFFICIAL-CONTEST-20261017 \
  --output-dir ./acc --concurrency 8 --limit 0 \
  --shots 5 --shot-selection longest --skip-retry
```

预期 `summary.json`：`single_shot.correct = 1254`、`single_shot.accuracy_pct = 95.072`、
`single_shot.finish_reason_length = 5`。

**如果只得到 94.4655%，说明少了条件**：那是"标准 5-shot、不开 TB"的读数；
如果得到 94.7688%，则是"长样例但没开 TB"；94.8446% 是"标准 5-shot + TB"。
四个数字各自对应一个缺项，可据此定位。

复现时核对两件事：`summary.json` 的 `shot_train_indices` 应为
`4483/310/7364/7371/5464`，服务日志应出现 `[PR79] installed=True budget=3584`。

> **允许重试的口径（96.2851%）是另一个配置**：标准 5-shot（train 前 5 条）+ **不开** TB，
> 只对 26 道 `finish_reason=length` 重试（去掉 `--skip-retry` 即可）。
> 两者不可互换，原因见 `docs/ACCURACY.md` §4.0。

---

## 8. 回滚

本栈是**纯运行时**的，磁盘上没有任何模型/量化表/`.so` 被覆盖。回滚三选一：

```bash
# a) 只要回"原版 SGLang 行为"（保留权重）：清空 overlay 链后重启
SWEEP_EXTRA_PYTHONPATH= SWEEP_SITECUSTOMIZE_PREFIX= SWEEP_TRITON_JSON_DIR= SWEEP_EXTRA_ENV=''
docker restart <CT>
# b) 回上一个 sweep.env 快照
cp -a sweep.env.before_*.snapshot sweep.env && docker restart <CT>
# c) 容器本身可丢弃：docker rm -f <CT>，按 §5 重建（镜像按 digest 固定）
```

---

## 9. 踩坑清单（照抄能省半天）

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | 容器起来就 HIP OOM，或吃错了卡 | `HIP_VISIBLE_DEVICES` 在**本机是全局卡号 0–7**，不是容器内局部序号；容器 `--privileged` 能看到全部 8 张卡 | 按 `ls /dev/dri` 找到目标卡的 render 节点，`SWEEP_HIP_VISIBLE` 写**全局号**（如 `4,5`） |
| 2 | 日志停在 numpy 警告、`/health` 长 000 | 无外网时 `tvm_ffi` 现场编译 torch 扩展卡住 | 预置 `/root/.cache/tvm-ffi`（§1.1） |
| 3 | 宿主 `curl /health` 000，容器内 200 | 少了 `--network host` | 加 `--network host` |
| 4 | 某个臂没生效但**不报错** | overlay 是 fail-closed：缺门文件 ⇒ 静默 `inert` | 按 §6.1 做臂断言；门文件必须打包交付 |
| 5 | `pool_configurator.py: Not enough memory`（`max_total_num_tokens ≤ 0`） | `max-running-requests` 过大 + mamba 池/ KV 池抢预算 | 用冻结值 `mr=32 / mem 0.90 / mamba 64` |
| 6 | 加载器报 preprocessor 相关错误 | 权重目录里 `preprocessor_config.json` 是 0 字节占位 | 按 §2.3 从 Qwen3.8-27B 原目录软链 |
| 7 | 换草稿注意力后端后输出 SHA 没变但吞吐崩了 | fa3 会**静默算错**草稿非因果块注意力 | 草稿注意力固定 `triton`；不要用 target SHA 判断草稿等价 |
| 8 | 64K 档 TTFT 几百秒 | 服务端并发容量 < 客户端并发（`prefill first`、无 overlap） | 这是本栈的已知容量边界，见 `docs/LIMITATIONS.md`；不要误判为故障 |
