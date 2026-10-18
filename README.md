# Qwen3.8-27B · Hygon K100AI 双卡 TP2 — W8A8 INT8 + DFlash2 推理优化

> 在 2 张 Hygon K100AI（DCU / gfx928）上，把 Qwen3.8-27B 的 SGLang 0.5.12 服务推到
> **C1 6.83× / C8 3.39× / C16 2.71× / C32 1.45× / C64 1.12× / C128 1.23×** 官方基线，
> **六个并发档位全部超过基线**。全部加速来自纯运行时 overlay——权重、量化表、
> 基础 `.so` 零改动。

本仓库给出三样东西：

1. **成绩**：按官方协议（2 轮预热 + 3 轮正式）跑出的全部数字与原始记录（§4、`results/`）。
2. **优化内容**：每一处加速改了什么、为什么有效、代价是什么（§2、`docs/ARCHITECTURE.md`）。
3. **复现**：从官方镜像 + 官方权重出发，一步步重建并自检（§3、`docs/REPRODUCE.md`）。

---

## 0. 核心成绩

### 性能 · 官方协议 4K 输入 / 1K 输出

| 并发 | 官方基线 tok/s | 本方案 tok/s | 相对基线 | 相对未优化 |
|---:|---:|---:|---:|---:|
| 1 | 9.055 | **61.81** | **6.83×** | 持平（−0.7%） |
| 8 | 52.480 | **177.87** | **3.39×** | **+58.6%** |
| 16 | 74.011 | **200.80** | **2.71×** | **+57.5%** |
| 32 | 142.864 | **207.31** | **1.45×** | **+64.5%** |
| 64 | 194.387 | **217.88** | **1.12×** | **+68.5%** |
| 128 | 181.599 | **223.00** | **1.23×** | **+67.2%** |

64K 长上下文（最高 **2.55×**）与 ShareGPT 真实对话流（最高 **6.85×**）见 §4。

### 精度 · GSM8K 全量 1319 题

| 口径 | 准确率 | 门槛 95.0 |
|---|---:|:--:|
| C8 并发 · 单次请求 | **95.07%**（1254/1319） | ✅ |
| C8 并发 · 允许重试 | **96.29%**（1270/1319） | ✅ |
| C1 单流 | **96.21%** | ✅ |
| C2 并发（前 300 题） | **95.67%** | ✅ |

四个口径全部达标。逐题记录、评测脚本与复现命令见 §5 与 `docs/ACCURACY.md`。

---

## 1. 方案架构

```
官方 Qwen3.8-27B  →  SmoothQuant W8A8-INT8 量化副本（权重文件不改动）
                  +  Qwen3.8-27B-DFlash2 五层 BF16 草稿模型（DFLASH 投机，draft 8 / step 1）
                  +  11 组运行时 overlay（Triton 调优表、FASTHEAD、CANDHEAD、SPLITKV、TAILSYNC、
                                          GDNFUSE、PREFILL_GDN、Q8B8 + 自研 HIP 扩展、ATTNCFG…）
                  +  启动参数 cap16（mem-fraction 0.90 / mamba pool 64 / cuda-graph-max-bs 16）
  SGLang 0.5.12 · TP2 · 2× K100AI
```

**没有任何权重被改写**：加速全部来自运行时 monkey-patch（overlay）、Triton/rocBLAS 调优表，
以及一个只做 `assign_fixed8_i64` 的自研 HIP 内核（bs ≤ 16）。回滚只需摘掉 overlay 链。

---

## 2. 优化内容

### 2.1 并发验证路径：高并发吞吐 +57% ~ +68%

本轮最大的一处收益。原栈在 `bs≥3` 时，q=8 的 target verify 会掉回较慢的 parent varlen
路径（`q8_batch2_v6` 里 `batch in (1,2)` 的守卫把它挡住了）。把守卫放宽到 `batch 1..32`、
让 verify 全程走 raw `paged_attention` 之后：

| 档位 | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| C8 | 112.15 | **177.87** | **+58.6%** |
| C16 | 127.57 | **200.80** | **+57.5%** |
| C32 | 126.06 | **207.31** | **+64.5%** |
| C64 | 129.31 | **217.88** | **+68.5%** |
| C128 | 133.35 | **223.00** | **+67.2%** |

同一处改动还修掉三处 `get_last_loc` 的 int64→int32 步长错位，使 bs≥3 的页表索引与
batch=1/2 对齐，高并发下的数值偏差随之消失。C1 不经过该路径，实测 61.81 vs 62.26
（−0.7%，轮次波动内），符合预期。

实现：`overlays/concurrency_fix_overlay`（`exact_get_last_loc` + `common.hcu_get_last_loc`
锚定 + q8 verify 放宽到 `batch 1..32`）。

### 2.2 W8A8 INT8 与 308 张 Triton 调优表

权重与激活都是 SmoothQuant W8A8-INT8。GPU 侧瓶颈不在 GEMM 本身，而在每个形状第一次
走到时 Triton 的 autotune 选择——本仓库直接带上 **308 张按 shape 命中的调优表**
（`overlays/tables/`），覆盖 M4/K8704/N5120 等热点形状，冷启动即走最优 kernel，不依赖
运行时 autotune 的运气。配套 `q8b8_overlay` 与 `gdn_fuse_overlay` 把 down-projection
的 DirectLDS 与 gate/up 融合固定下来（M4/K8704/N5120 精确匹配才生效，否则自动回退）。

### 2.3 DFlash2 投机解码

五层 BF16 草稿模型，`draft 8 tokens / 1 step`，target verify 与草稿前向在同一次
CUDA graph 里完成。关键取舍：

* **草稿注意力固定 `triton`**：换 fa3 后输出 SHA 不变但吞吐崩，因为 fa3 会静默算错
  草稿的非因果块注意力。这条写进 `docs/REPRODUCE.md` 的踩坑清单。
* **SPLITKV 16 路滑窗切分**（`overlays/draft_splitkv_overlay`）：把草稿 KV 按 16 段切，
  减少单次读放大。
* **TAILSYNC 去同步**（`overlays/tail_overlay`）：去掉尾部冗余同步点。
* **FASTHEAD / CANDHEAD**：把草稿的候选头与词表头按档位钉解，避免每步重算。
* **自研 `assign_fixed8_i64` HIP 内核**（`native_ext/`）：batch>1 的 DFlash prepare
  用单波 int64 偏移分配，替代原来多次 kernel launch 的写法。

### 2.4 11 组运行时 overlay：fail-closed、可单独摘除

每组 overlay 都是一个 `sitecustomize.py` + 一个实现文件，只做 monkey-patch，不落盘改源码。
它们**fail-closed**：门文件（`overlays/gates/*_cfg.txt`）缺失时该组静默 `inert`，不会
半生效。因此可以逐组开关做消融，也可以整体摘掉回到官方行为（`docs/REPRODUCE.md` §8）。

### 2.5 64K 长上下文

长输入下时间几乎全花在 prefill/TTFT 上（C32 的 TTFT 就有 1256 s），所以这一档的优化
集中在 prefill 侧：PREFILL_GDN 移植、`chunked-prefill 16384`、`mamba-track 256`，
以及长 KV 的 BM128 q8 verifier。结果 C1 19.20（2.55×）、C8 23.72（1.70×）、
C16 23.50（1.65×）、C32 24.33 tok/s。

---

## 3. 快速开始

```bash
# 0) 前置：2 张 K100AI（DCU gfx928）、能拉取官方 SGLang 镜像、本地有官方 W8A8 权重与 DFlash2 草稿
# 1) 校验并准备目录（会逐项校验 SHA256，缺一个就中止）
bash scripts/setup_release.sh
# 2) 拉起服务（等价于本次实测的容器）
bash scripts/entry_official_contest.sh
# 3) 两道自检门：C1 逐位确定性锁 + 20 题质量门
# 4) 官方协议全量测试
bash scripts/run_official_suite_official.sh
```

详细步骤、每一条命令的期望输出、以及「错了怎么排查」见 `docs/REPRODUCE.md`。

---

## 4. 性能成绩（详细）

官方协议（`/v1/completions`，4096 输入 / 1024 输出，固定随机 token id，seed 20260912，
每轮 `max(8, 2×C)` 请求）下的输出吞吐对比官方基线。

**下表为合入并发路径优化后的成绩**——`bs>=3` 的 q=8 verify 从 parent varlen 切到
raw `paged_attention`，高并发档位整体提升 **+57% ~ +68%**。优化前的对照数据见
`docs/RESULTS_OFFICIAL.md` §2.1。

> **全部六个档位都已按官方标准 2 预热 + 3 正式轮复测**（`perf_std_*`），主列即该口径。
> 1+2 轮那列只作补充，通常比标准口径高 0.7%~6.7%（C1 因每轮仅 8 请求、波动大而反向 +4.8%）。

<!-- RESULTS_TABLE_START -->
### 4K input / 1K output

| C | 官方基线 | **标准 2+3 轮** | 相对基线 | 1+2 轮（补充） | 优化前 | 提升 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.055 | **61.81** | **6.83x** | 59.0 | 62.26 | −0.7%（持平） |
| 8 | 52.480 | **177.87** | **3.39x** | 190.12 | 112.15 | **+58.6%** |
| 16 | 74.011 | **200.80** | **2.71x** | 208.90 | 127.57 | **+57.5%** |
| 32 | 142.864 | **207.31** | **1.45x** | 222.12 | 126.06 | **+64.5%** |
| 64 | 194.387 | **217.88** | **1.12x** | 221.02 | 129.31 | **+68.5%** |
| 128 | 181.599 | **223.00** | **1.23x** | 224.52 | — | — |

**主列是标准 2 预热 + 3 正式轮**（与官方协议一致）；1+2 轮作为补充，数值通常高 4~6%。
标准口径下 C8/C16/C32/C64/C128 分别是官方基线的
**3.39x / 2.71x / 1.45x / 1.12x / 1.23x**，
比未修复链高 **+58.6% / +57.5% / +64.5% / +68.5%**（C128 修复前 133.35，现 223.00，**+67.2%**）。
标准轮次比 1+2 轮低 0.7%~6%，与轮次差异的预期一致。

> **C1 现在有同口径对照了**：标准 2+3 轮测得 **61.81**，修复前同样是 2+3 轮的 62.26，
> 差 −0.7%，落在 C1 自身的轮次波动内（3 轮 min 50.67 / max 69.30，每轮只有 8 个请求）。
> 这符合预期——C1 只跑单请求，不走本次修复触及的 `bs>=3` 路径。1+2 轮的 59.0 是
> 更保守的补充读数，不再用于对比。
>
> **C32/C64/C128 现在全部反超官方基线**——修复前 C32 只有 0.88x（唯一落后档位）。
> 标准口径下 C32 = 1.45x，结论不变。

### 64K input / 1K output

| C | 官方基线 tok/s | 修复后 tok/s | 相对基线 | 说明 |
|---:|---:|---:|---:|---|
| 1 | 7.520 | **19.20** | **2.55x** | TTFT 29.5 s，TPOT 26.75 ms |
| 8 | 13.929 | **23.72** | **1.70x** | TTFT 268 s（长输入由 prefill 主导） |
| 16 | 14.242 | **23.50** | **1.65x** | TTFT 622 s |
| 32 | — | **24.33** | — | TTFT 1256 s，TPOT p95 124 ms；官方基线未给出该档 |

> **这一档不要用"提升"的直觉读**：64K 的四个档位修复后都比未修复链**略低**
> （C1 19.76→19.20、C8 24.11→23.72、C16 24.93→23.50、C32 24.92→24.33，−1.6%~−5.7%）。
> 原因不是修复有害，而是**长输入下时间几乎全花在 prefill/TTFT 上**——C32 的 TTFT 就有
> 1256 秒，decode 阶段在整轮时间里占比极小，而本次修复只改 decode 的 verify 路径。
> 相对官方基线的 1.65~2.55x 来自本栈整体（W8A8 + DFlash2 + 调度），不是这一次修复。

### ShareGPT

| C | 官方基线 tok/s | 本方案 tok/s | 相对基线 | 状态 |
|---:|---:|---:|---:|---|
| 1 | 12.150 | **30.88** | **2.54x** | PASS（1000/1000） |
| 4 | 20.260 | **138.88** | **6.85x** | PASS（1000/1000） |
| 16 | 104.905 | **210.73** | **2.01x** | PASS（1000/1000） |
| 32 | 168.419 | **210.85** | **1.25x** | PASS（1000/1000） |
| 64 | 242.089 | **206.58** | 0.85x | PASS（1000/1000），受 slot 限制 |

> **这一组的 C16/C32/C64 是"mamba slot 上限"下的读数，不能当算力上限看**：
> ShareGPT 跑在 18636，实测启动参数是 `--max-running-requests 32`
> `--max-mamba-cache-size 16` `--mem-fraction-static 0.90` `--chunked-prefill-size 16384`
> ——与 4K/64K 矩阵**同一套配置**。但 ShareGPT 的请求远长于固定长度矩阵，每个请求
> 都要占一份 mamba 状态，**16 个 mamba slot 先于 32 个 request slot 打满**，所以
> 有效并发被压在约 16。
>
> 硬证据是 **C16 = 210.73 与 C32 = 210.85 几乎完全相同**：并发从 16 加到 32，吞吐
> 一点没涨，说明瓶颈是 mamba slot 而不是算力。C64 因此低于官方基线（0.85x），
> 这是长上下文对话流的容量边界，不是本方案变慢——同样修复版在 4K/1K 矩阵上 C64
> 是基线的 1.12x（那批请求短，mamba slot 不是瓶颈）。**C1 与 C4 未触到该上限**，
> 2.54x 与 6.85x 是可信的（C1 的 TTFT 339 ms、TPOT 35.24 ms 也说明单请求没被排队污染）。
> 详细说明见 `docs/RESULTS_OFFICIAL.md` §1.2。
<!-- RESULTS_TABLE_END -->


## 5. 精度成绩与复现

GSM8K 全量 1319 题，评测脚本与逐题原始输出都在 `results/`：

| 口径 | 准确率 | 说明 |
|---|---:|---|
| C8 并发 · 单次请求 | **95.07%**（1254/1319） | 长样例 5-shot + Thinking Budget 软闭合 |
| C8 并发 · 允许重试 | **96.29%**（1270/1319） | 标准 5-shot，仅对截断题重试 |
| C1 单流 | **96.21%** | 单请求无 batch 内数值差异 |
| C2 并发（前 300 题） | **95.67%** | |
| 0-shot 单次（参考） | 90.90% | 不用 few-shot 的对照 |

复现命令（两条路径任选，对应上表前两行）：

```bash
# 单次请求口径
python3 scripts/gsm8k_retry_pipeline.py --shots 5 --shot-selection longest --skip-retry

# 允许重试口径
python3 scripts/gsm8k_retry_pipeline.py --shots 5
```

其他基准（RealWorldQA、LCB v6、MathVision）的口径与结果见 `docs/ACCURACY.md`。

---

## 6. 仓库结构

```text
docs/
  REPRODUCE.md        从原始镜像 + 权重复现本方案的完整步骤与自检门
  ARCHITECTURE.md     11 组 overlay 逐个说明（改了什么、为什么、代价/边界）
  METHODOLOGY.md      方法论与工程取舍
  RESULTS_OFFICIAL.md 官方协议成绩（4K/1K、64K/1K、ShareGPT）+ 逐档原始记录
  ACCURACY.md         精度门（GSM8K / RealWorldQA / LCB v6 / MathVision）
  LIMITATIONS.md      适用范围与技术约束
manifests/
  quantization_manifest.json   量化契约（权重/激活/KV/草稿/head/原生扩展及其 SHA）
  run_manifest.json            运行契约（镜像 digest、参数、挂载、数据集 SHA、seed）
  accuracy_results.json        精度逐项结果
  SHA256SUMS                   交付物逐文件校验和（398 条）
overlays/                     运行时 overlay 源码 + 开关门文件 + W8A8 Triton 调优表
runtime_patches/              17 个 bind-mount 运行时补丁（与容器挂载一一对应）
native_ext/                   assign_fixed8_i64（HIP 源码 + 编译出的 .so）
scripts/                      一键脚本、官方 benchmark、精度脚本、确定性/质量探针
results/                      原始结果（summary.json / requests.jsonl / 日志 / 证据）
```

---

## 7. 适用范围

* 本方案是**运行时 overlay 栈**，不是重编译的 SGLang fork。可复现性依赖三样东西：
  **官方镜像 digest**、**17 个运行时补丁文件**、**overlay + 调优表**——仓库全部自带。
* 硬件前提是 **2 张 K100AI（gfx928）做 TP2**；单卡或 4 卡需要重新标定 mamba/KV 预算。
* 并发上限由 `mamba_pool` 决定：本配置 `max-running-requests=32`、`max-mamba-cache-size=16`，
  固定长度矩阵下 C64/C128 靠排队仍可跑满（实测 217.88 / 223.00 tok/s）；
  但**长对话流**（ShareGPT）每个请求占一份 mamba 状态，16 个 slot 会先打满，
  因此 C32/C64 的读数受该上限约束（详见 §4 ShareGPT 注）。
* 本仓库**不含**模型权重（体积大且受各自许可约束），只含"如何得到它们"的说明与校验和。

---

## 8. 许可与致谢

* 本仓库自有代码（overlay、脚本、内核）以 **Apache-2.0** 发布，见 `LICENSE`。
* SGLang 及其镜像、Qwen3.8-27B 权重、DFlash2 草稿模型各自遵循其原始许可。
* 上游工作：SGLang（Apache-2.0）、Qwen（Apache-2.0）、Hygon K100AI 平台运行时。
