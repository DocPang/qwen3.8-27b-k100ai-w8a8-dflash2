# 结构：这套方案到底改了什么

本方案的性能提升全部发生在**运行时**。它由三层组成：

1. **调优表**（纯调度参数，数值逐位不变）；
2. **overlay 臂**（monkey-patch 上游 Python 路径，逐臂可单独开关、fail-closed）；
3. **一个自研 HIP 内核**（`assign_fixed8_i64`，把槽位映射内核的 batch 上限从 8 扩到 16）。

> **路径约定**：所有 overlay 都硬编码 `_EXP=/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013`。
> 复现时 `scripts/setup_release.sh` 会把本仓库的 `overlays/` 铺到这个路径（或在同一路径下放软链），
> 这样上游源码一行都不用改。若你想换根目录，需要把每个 overlay 顶部的 `_EXP` 常量改掉（每臂一处）。

---

## 1. 硬件与软件栈

| 项 | 值 |
|---|---|
| 设备 | 2 × K100AI（DCU，`gfx928`，120 CU） |
| 并行 | TP2，`--pp-size 1` |
| 框架 | SGLang 0.5.12（SourceFind/Hygon 镜像，digest 固定） |
| 目标模型 | 官方 Qwen3.8-27B → SmoothQuant W8A8-INT8 |
| KV | BF16，`--page-size 64`，`--context-length 262144` |
| 投机 | DFLASH，5 层 BF16 草稿，`num_steps=1`，`num_draft_tokens=8` |
| 注意力后端 | target `fa3`，**draft `triton`**（必需，见 `REPRODUCE.md` §2.2） |
| graph | `--cuda-graph-max-bs 16`，`--disable-piecewise-cuda-graph` |
| 内存 | `--mem-fraction-static 0.90`，`--max-mamba-cache-size 64`（cap16），`--max-running-requests 32` |

---

## 2. 调优表：W8A8 Triton（`overlays/tables/`，308 个 json）

* 作用对象：`lightop` 的 W8A8 INT8 GEMM 分块调度（`BLOCK_M/N/K`、`SPLIT_K`、`num_warps`、`num_stages`）。
* 做法：按 `(N,K)` 离线扫参，把每个形状的最优配置**钉死**成 json；运行时通过 `TRITON_JSON_DIR` 注入。
* 数值风险：**零**。所有改动桶都取 `SPLIT_K=1`，即只改分块调度、不改累加顺序 ⇒ 逐位一致。
* 代价：单步省约 **5.2 ms**（在 M=8 的 target 前向上测得；相对当时基线）。
* 复现注意：表是**按形状**索引的，与权重来源无关，因此同一形状的任何 Qwen3.8-27B 变体都能用。

---

## 3. overlay 臂（逐臂说明）

链序（自外向内）：`candhead_overlay → combo2_overlay → tail_overlay → (gdn_fuse / prefill_gdn)`
以及并列的 `head_overlay`、`q8b8_overlay`、`attncfg_overlay`、`draft_splitkv_overlay`。
每个臂都是 `runpy` 先加载父层再替换目标函数，并且**全部 fail-closed**：
缺门文件即静默 `inert`（只打印一行日志，不报错）——所以门文件必须随发布物一起交付。

| 臂 | 改的地方 | 机制 | 代价/边界 |
|---|---|---|---|
| **FASTHEAD**<br>`head_overlay` | `LogitsProcessor._compute_lm_head` | 厂商只给了 M=1(10347)/M=4(11055) 的 rocBLAS 显式解；verifier 的 head 跑在 M=8×bs，默认 dispatch 只有 308 GB/s(M=8)/97 GB/s(M=32)。本臂把 M=8..128 全档位钉到最优解（≈575–677 GB/s） | 非逐位（换 rocBLAS 解），离线 480/480 argmax 一致；`head64_sids.txt` 给出 M→solution 表 |
| **CANDHEAD**<br>`candhead_overlay` | `models/dflash.py:compute_candidates` 里的 `torch.matmul(hidden, W[:num_org].T)` | 这个 (M,5120)×(5120,124160) bf16 GEMM 的 M = bs×7，默认 rocBLAS 在 M≥21 时塌方（9.4–30 ms）；钉解后 M=56(bs8) **19.9→2.38 ms（−88%）** | 只替换这一个 matmul，topk/all_gather 逐行保留；有 top-16 不变性检查；**M>70（bs>10）翻转 tie-band，因此只在 M≤70 生效** |
| **ATTNCFG**<br>`attncfg_overlay` | `_get_block_sizes_for_extend_attention` | 草稿 decode 的 extend attention 只有 16 个 block（120 CU 里 104 个空闲），实测 KV 带宽 13.7 GB/s（上限 682）；把 decode 路径 tile 换成 `16 64 8` | **只在 `max_len_extend ≤ 64`（decode）时替换**，prefill 路径（4096）保持厂商 tile ⇒ TTFT/prefill 构造性不回归 |
| **SPLITKV**<br>`draft_splitkv_overlay` | 草稿 extend attention 的 `_fwd_kernel` | 克隆 kernel 增加 KV 方向切分坐标：grid 从 `(batch, head, cdiv(Lq,BM))` 变成 `(batch, head, cdiv(Lq,BM)×NUM_SPLITS)`，partial (acc,m,l) 再用 logsumexp 合并。`NUM_SPLITS=1` 与厂商循环结构逐位一致（零风险 smoke） | `NUM_SPLITS=16` 改变 online-softmax 分组 ⇒ **不假设逐位**；用 accept + 内容 sha 把关；单步省约 1.9 ms |
| **TAILSYNC**<br>`tail_overlay` | `triton_ops/fused_kv_materialize.py::materialize` | 每步开头的 `int(positions.max().item())` 是一次 GPU→CPU 同步（cProfile 测到 8 次/0.050 s ≈ 6.25 ms/次，tail 中最大的 host 阻塞）。两次用途都只是"边界安全检查"，把 RoPE cache 一次性扩到 `context-length` 之后该检查永不触发 | **第一次调用完全走厂商路径**（同步/扩容/校验各发生一次），之后走无 `.item()` 的副本；数学不变 ⇒ 期望数值惰性 |
| **GDNFUSE**<br>`gdn_fuse_overlay` | `models/qwen3_5.py` GDN in-proj | 该 checkpoint `num_v_heads/num_k_heads = 48/16 = 3`，不在厂商融合内核的 `[1,2,4]` 白名单里 ⇒ 48 层 GDN 全走慢分支：每层 2 次 `contiguous()` + 1 次 `torch.cat` ≈ **144 次发射/步**。本臂用自研 Triton 核把慢分支合成 1 次发射 | 纯数据搬运（S∈{1,8,16,32} 逐位验证）；`z` 返回 `(S,nv_tp,head_v)` 视图以保持 `z_shape_og`；对 `qwen3_5.py` 做 md5 钉解 |
| **PREFILL_GDN**<br>`prefill_gdn_overlay` | `sglang.srt.layers.attention.fla.chunk` | 把生产线的 token-major GDN prefill 加速栈搬过来：用 `sys.meta_path` finder 对**单个 dotted name** 提供补丁版 `chunk.py`（不动容器文件系统），并前置 AITER backport 路径 + `gdn8192` 精确预填 tile 钉解 | 纯加法补丁（对照文件与安装版字节相同）；缺任一 `/data` 路径即 inert |
| **Q8B8**<br>`q8b8_overlay` | raw-q8 verifier 快路径 | 上游只对 `batch ∈ {1,2}` 走单次 `paged_attention` 裸调用，`batch ≥ 3` 落回 varlen 父路径；C8 并发时 verify batch=8，于是 16 层全注意力在**最关键的时刻**全走慢路径。本臂把 batch 范围扩到 `1..8`（裸调用与 v6 逐字一致，含 ABI 额外 int 0） | 门限是**有限的**：曾试扩到 bs≤12，accept-free step 在 C8/C16/C32 全面退化（+12.1%/+6.3%/+8.3%）⇒ **bs≤8 才是正确门限** |
| **SPLITKV 门** | — | 见上 | — |
| **COMBO2**<br>`combo2_overlay` | 组合臂 | 把"保留候选 1（decode：GDNFUSE）"和"保留候选 2（prefill：PREFILL_GDN）"合成**一个**臂；因为两个子臂都会各自链一遍 `tail_overlay`，组合时临时把 `runpy.run_path` 换掉、让重复的 tail 链变 no-op，避免补丁被套两次 | 子臂各自保留门文件/SHA 钉解/fail-closed；一键可回退 |
| **COMBO**<br>`combo_overlay` | 早前的组合臂（草稿候选头 + decode tile） | 已被 CANDHEAD/ATTNCFG 取代，保留在仓库里作为可追溯历史 | 不作为发布配置启用 |
| **ATTN_LAUNCH**<br>`attn_launch_overlay` | `num_stages` / `waves_per_eu` | 扫描 launch 配置 | **已证据关闭**：`ns=2 +3.8 ms`、`ns=3 +3.7 ms`（数值完全相同，纯调度），不启用 |

### 3.1 门文件（`overlays/gates/`）

| 门文件 | 内容 | 含义 |
|---|---|---|
| `candhead_cfg.txt` | `1` | CANDHEAD 开 |
| `combo2_cfg.txt` | `1` | COMBO2 开 |
| `gdn_fuse_cfg.txt` | `1` | GDNFUSE 开 |
| `prefill_gdn_cfg.txt` | `1` | PREFILL_GDN 开 |
| `q8b8_cfg.txt` | `1` | Q8B8 开 |
| `attn_cfg.txt` | `16 64 8` | ATTNCFG：BLOCK_M BLOCK_N num_warps |
| `draft_splitkv_cfg.txt` | `16` | SPLITKV：NUM_SPLITS |
| `tail_sync_cfg.txt` | *空文件* | TAILSYNC：**存在即开**（内容必须为空，非空会被启动脚本判为 BAD） |

> **发布纪律**：探针类 overlay（`chainprobe`、`specprobe` 等）不属于发布物，
> 其门文件必须**不存在**。`scripts/setup_release.sh` 会在启动前检查这一点。

---

## 4. 自研 HIP 内核：`assign_fixed8_i64_c16exp_v2`

* 位置：`native_ext/`（`.hip` 源码 + `.so`，sha256 `99ef8a6a…09414`）。
* 作用：`assign_fixed8_i64` 是每步都要跑的"请求 → KV 槽位"映射内核。
  上游实现只支持 `batch_size ≤ 8`；本内核用 64 段展开（`c16exp`）把它扩到 **`batch ≤ 16`**。
* 正确性：**64-trial 逐元素离线门**（`bs = 1..16` 全 PASS）+ 装载器运行时 AUDIT。
  在"是它导致 C1 锁漂移"的怀疑被提出后，这一臂被单独做了解除审查（见 `METHODOLOGY.md` §4）。
* 边界：这是**唯一**一个改数值路径的自研内核；它只改**哪个槽位被分配到哪个请求**，
  不参与前向计算，因此只影响调度/内存布局，不影响 logits。

---

## 5. 与镜像的关系（为什么叫"运行时 overlay"）

```
官方镜像（by digest）
  └─ sglang 0.5.12 安装树
       ├─ 16 个文件被 bind-mount 替换（runtime_patches/）      ← 项目自己的运行时补丁
       ├─ 1 个目录被 bind-mount 覆盖（kernels/ops/speculative） ← DFlash2 selector kernel
       └─ sitecustomize 链（overlays/）                        ← 11 组猴补丁
         + TRITON_JSON_DIR 调优表
         + SGLANG_Q38_ASSIGN_FIXED8_I64_SO 自研内核
```

**磁盘上的权重、量化表、镜像内的 `.so` 本体都没有被改写**。
这也意味着：把 overlay 链摘掉（4 个环境变量清空 + 重启）就回到原版 SGLang 行为。
