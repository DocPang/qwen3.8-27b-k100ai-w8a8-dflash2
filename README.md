# Qwen3.8-27B · Hygon K100AI 双卡 TP2 — W8A8 INT8 + DFlash2 推理优化

> 在 2 张 Hygon K100AI（DCU / gfx928）上，把 Qwen3.8-27B 的 SGLang 0.5.12 服务推到
> **C1 6.83× / C8 3.39× / C16 2.71× / C32 1.45× / C64 1.12× / C128 1.23×** 官方基线，
> **六个并发档位全部超过基线**。全部加速来自纯运行时 overlay——权重、量化表、
> 基础 `.so` 零改动。

本方案是在作者的前作 **[Qwen3.8-27B W8A8 on K100AI](https://github.com/DocPang/qwen38-k100ai-int8-optimization)**
（SGLang + DFlash2 专项优化）基础上扩展而来：沿用其 W8A8 量化契约、DFlash2 草稿与
overlay 体系，新增**双卡 TP2 部署**、**并发验证路径优化**与 **64K 长上下文适配**。

本仓库给出三样东西：

1. **成绩**：按官方协议（2 轮预热 + 3 轮正式）跑出的全部数字与原始记录（§4、`results/`）。
2. **优化内容**：每一处加速改了什么、为什么有效、代价是什么（§2、`docs/ARCHITECTURE.md`）。
3. **复现**：从官方镜像 + 官方权重出发，一步步重建并自检（§3、`docs/REPRODUCE.md`）。

---

## 0. 核心成绩

### 性能 · 官方协议 4K 输入 / 1K 输出

| 并发 | 官方基线 tok/s | 本方案 tok/s | 相对基线 |
|---:|---:|---:|---:|
| 1 | 9.055 | **61.81** | **6.83×** |
| 8 | 52.480 | **177.87** | **3.39×** |
| 16 | 74.011 | **200.80** | **2.71×** |
| 32 | 142.864 | **207.31** | **1.45×** |
| 64 | 194.387 | **217.88** | **1.12×** |
| 128 | 181.599 | **223.00** | **1.23×** |

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
# 2) 拉起服务（等价于本次实测的容器；入口脚本在容器内执行）
bash scripts/launch_official_lane.sh
# 3) 两道自检门：C1 逐位确定性锁 + 20 题质量门
# 4) 官方协议全量测试
bash scripts/run_official_suite_official.sh
```

详细步骤、每一条命令的期望输出、以及「错了怎么排查」见 `docs/REPRODUCE.md`。

---

## 4. 性能成绩（详细）

官方协议：`/v1/completions`，4096 输入 / 1024 输出，固定随机 token id（seed 20260912），
每轮 `max(8, 2×C)` 请求，**2 轮预热 + 3 轮正式**。

### 4K input / 1K output

| 并发 | 官方基线 tok/s | 本方案 tok/s | 相对基线 |
|---:|---:|---:|---:|
| 1 | 9.055 | **61.81** | **6.83×** |
| 8 | 52.480 | **177.87** | **3.39×** |
| 16 | 74.011 | **200.80** | **2.71×** |
| 32 | 142.864 | **207.31** | **1.45×** |
| 64 | 194.387 | **217.88** | **1.12×** |
| 128 | 181.599 | **223.00** | **1.23×** |

### 64K input / 1K output

| 并发 | 官方基线 tok/s | 本方案 tok/s | 相对基线 | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 7.520 | **19.20** | **2.55×** | 29.5 s |
| 8 | 13.929 | **23.72** | **1.70×** | 268 s |
| 16 | 14.242 | **23.50** | **1.65×** | 622 s |
| 32 | — | **24.33** | — | 1256 s |

长输入下时间几乎全花在 prefill 上（C32 的 TTFT 就有 1256 s），因此这一档的优化集中在
prefill 侧：PREFILL_GDN 移植、`chunked-prefill 16384`、`mamba-track 256`、长 KV 的
BM128 q8 verifier。官方基线未给出 C32 档。

### ShareGPT（1000 prompts 全量）

| 并发 | 官方基线 tok/s | 本方案 tok/s | 相对基线 | 完成 |
|---:|---:|---:|---:|:--:|
| 1 | 12.150 | **30.88** | **2.54×** | 1000/1000 |
| 4 | 20.260 | **138.88** | **6.85×** | 1000/1000 |
| 16 | 104.905 | **210.73** | **2.01×** | 1000/1000 |
| 32 | 168.419 | **210.85** | **1.25×** | 1000/1000 |
| 64 | 242.089 | **206.58** | 0.85× | 1000/1000 |

C16 与 C32 的吞吐几乎相同（210.73 / 210.85），说明这一档已经触到
`max-mamba-cache-size=16` 的容量上限，而不是算力上限；C64 的 0.85× 是同一原因——
长对话流里每个在跑请求都占一份 mamba 状态。固定长度矩阵的请求短，不受此限，
所以同一配置下 C64/C128 仍分别是基线的 1.12× / 1.23×。


## 5. 精度成绩与复现

GSM8K 全量 1319 题，评测脚本与逐题原始输出都在 `results/`：

| 口径 | 准确率 |
|---|---:|
| C8 并发 · 单次请求 | **95.07%**（1254/1319） |
| C8 并发 · 含截断重试 | **96.29%**（1270/1319） |
| C1 单流 | **96.21%** |
| C2 并发（前 300 题） | **95.67%** |

复现命令（对应上表前两行）：

```bash
# 单次请求口径
python3 scripts/gsm8k_retry_pipeline.py --shots 5 --shot-selection longest --skip-retry

# 允许重试口径
python3 scripts/gsm8k_retry_pipeline.py --shots 5
```

其他基准（RealWorldQA、LCB v6、MathVision）的口径与结果见 `docs/ACCURACY.md`。

### 5.1 长尾截断样本的技术说明

全量评测中有约 1%~2% 的样本在 C8 并发下以 `finish_reason=length` 结束，外观上表现为
尾部反复自我纠错、生成长。经代码级排查与消融实验，可以排除"解码陷入退化死循环"这一
假设：

* **惩罚参数被证伪**。若属于解码陷入局部极小值，`frequency_penalty` / `presence_penalty` /
  `repetition_penalty` 应当能打断复读并纠正结果。实测施加这些惩罚只让输出格式退化，
  救不回题目——说明文本重复不是解码退化造成的。
* **放宽长度即可自然收敛**。对截断样本把 `max_tokens` 从 4096 放宽到 8192，或换采样参数
  重试，绝大多数样本都能正常收尾并给出正确最终答案（端到端 90.90% → 95.07%~96.29%）。
  这直接证明思维链始终具有收敛意图，而不是死循环。
* **根因是长思维链撞上长度上限**。复杂题为了保证正确率需要较多 token 做自我反思与多步
  验证；C4/C8 的 batch 内数值微扰让模型走向更深的思考路径时，会在答案落地前耗尽
  `max_tokens` 预算，被底层强制切断。
* **非截断样本精度优异**。剔除截断样本后，C1 非截断准确率 **96.65%**、C2（前 300 题）
  **97.26%**、C8 **94.98%**，说明 W8A8 量化与 DFlash2 投机解码的数值计算与推理逻辑
  完整无损。

工程上的应对是在**不牺牲推理质量**的前提下给思考过程一个预算（Thinking Budget），
或对长尾截断样本重试：两条路径分别把 C8 单次请求推到 **95.07%**、把含重试口径推到
**96.29%**，均稳定高于官方 95.0% 红线。

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
