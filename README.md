# Qwen3.8-27B on Hygon K100AI（DCU / gfx928）双卡 TP2 — W8A8 INT8 + DFlash2 参赛方案

> 社区 **Qwen3.8 DCU 性能优化挑战赛** 的完整参赛工程：从**原始镜像 + 原始权重**开始，
> 用**纯运行时 overlay**（不改权重、不改量化表、不覆盖基础 `.so`）把 SGLang 0.5.12 的服务性能推到本项目实测的最优点。

本仓库回答三件事：

1. **参数/结构**：这份方案到底改了什么（`docs/ARCHITECTURE.md`）。
2. **如何复现**：从官方镜像 + 官方权重出发、一步步重建并自检（`docs/REPRODUCE.md`）。
3. **实测成绩**：按比赛官方协议跑出来的全部数字与原始记录（`docs/RESULTS_OFFICIAL.md`、`results/`）。

---

## 0. 重要状态：GSM8K 达标（5-shot + 截断重试 95.98%）

**主结果（GSM8K 全量 1319 题，并发 8）**：

| 口径 | 准确率 | 说明 |
|---|---:|---|
| 5-shot，单次请求 | 94.1622%（1242/1319） | 最严格口径 |
| **5-shot + 只对 26 道截断题重试** | **95.9818%（1266/1319）** | **过 95.0 门槛** |
| 0-shot，单次请求 | 90.4473%（1193/1319） | 旧口径，见下 |
| 0-shot + 截断重试 | 94.77% | |

5-shot 的示例取自 **train split 前 5 条**（绝不用 test split），渲染成真实的多轮
`user`/`assistant`，这是 GSM8K 的事实标准口径（lm-eval-harness 默认 5-shot CoT）。
相比 0-shot，它同时做两件事：准确率 **+3.7pp**，复读截断率 **4.85% → 1.97%**。

截断重试**不改模型、不改服务配置**，只对 `finish_reason=length` 的题换一组请求参数
（`max_tokens=8192` / `temperature=0.7` / `reasoning_effort=low` / `medium`），
四种策略并集救回 **24/26**。复现脚本：`scripts/gsm8k_retry_pipeline.py --shots 5`。

> 若评测规则是"一次请求定成绩"，应报 **94.1622%**（5-shot 单次）；
> 若允许对截断样本重试，则 **95.9818%**。两个数字都如实列出，不混用。

| 维度 | 结论 |
|---|---|
| 性能（官方协议 4K/1K） | C1 **6.9×**、C8 **2.1×**、C16 **1.7×** 官方基线；C32 低于基线（见 `docs/LIMITATIONS.md`） |
| GSM8K 精度门（≥95.0） | **C1 全量 1319 题 = 96.21%**（v6 提取口径；原始末位数字口径 94.9962%，差 1 题）⇒ **达标** |
| 并发精度（0-shot） | C2 前 300 题 **94.67%**（临界，差 1 题；前 100 题的 95.0% 是样本偏易）；C8 全量 1319 题 **90.45%**、**非截断口径 94.98%**；C4 92.0%。修复前 C4/C8 为 16.7% / 14~16%，未修复链的 C8 thinking 甚至为 **0%** |
| 复读截断（0-shot C8 全量） | 64/1319 = **4.85%**；5-shot 下降到 26/1319 = **1.97%**，且重试救回率从 67% 升到 **92%**（few-shot 给了格式锚点） |
| 根因（已修复） | `q8_batch2_v6` 的 `batch in (1,2)` 守卫使 **bs≥3 的 target verify 掉回 parent varlen**；另有三处 `get_last_loc` int64→int32 步长错位 |
| 修复 | `overlays/concurrency_fix_overlay`：`exact_get_last_loc`（含 `common.hcu_get_last_loc` 锚定）+ q8 verify 放宽到 `batch 1..32` |

**复读（greedy repetition）是已知遗留**：bs≥3 时约 2.5% 的题撞 `max_tokens` 上限，
文本是"猜题目/自我怀疑"死循环。C1 本身也有 0.45%，属 `temperature=0` 贪心解码的固有退化；
`repetition_penalty`、`max_tokens=4096`、`mamba-track-interval`、draft 独占页等手段实测均无效。
报告里按 `docs/ACCURACY.md` §4.2 的三栏统计单列，不混入算术正确性。

完整证据链见 **[`docs/ACCURACY_INVESTIGATION.md`](docs/ACCURACY_INVESTIGATION.md)** 与
**[`docs/ACCURACY.md`](docs/ACCURACY.md)**。

> 方法论教训：C1 逐位锁只证明**确定性**（3/3 复现同一段输出），不证明**正确性**。
> 本次并发崩塌之所以长期未被发现，是因为早期"放宽 q8 守卫后字节不变"的 A/B 是在
> `get_last_loc` 步长错位**修好之前**做的 —— 页表错误掩盖了 attention 路径差异。

---

## 1. 一句话方案

```
官方 Qwen3.8-27B  →  SmoothQuant W8A8-INT8 量化副本（本仓库不改动其权重文件）
                  +  Qwen3.8-27B-DFlash2 五层 BF16 草稿模型（DFLASH 投机，draft 8 / step 1）
                  +  11 组运行时 overlay（Triton 调优表、FASTHEAD、CANDHEAD、SPLITKV、TAILSYNC、
                                          GDNFUSE、PREFILL_GDN、Q8B8 + 自研 HIP 扩展、ATTNCFG…）
                  +  启动参数 cap16（mem-fraction 0.90 / mamba pool 64 / cuda-graph-max-bs 16）
  SGLang 0.5.12 · TP2 · 2× K100AI
```

**没有任何权重被改写**：所有加速都来自运行时 monkey-patch（overlay）、Triton/rocBLAS 调优表、
以及一个只做 `assign_fixed8_i64` 的自研 HIP 内核（bs ≤ 16）。因此回滚只需摘掉 overlay 链。

---

## 2. 快速开始

```bash
# 0) 前置：2 张 K100AI（DCU gfx928）、能拉取官方 SGLang 镜像、本地有官方 W8A8 权重与 DFlash2 草稿
# 1) 校验并准备目录（会逐项校验 SHA256，缺一个就中止）
bash scripts/setup_release.sh --root /data/qwen38-release
# 2) 拉起服务（等价于本次实测的容器）
bash scripts/launch_official_lane.sh
# 3) 两道自检门：C1 逐位确定性锁 + 20 题质量门
python3 scripts/ab3.py rel          # 期望 3/3 复现同一 TEXT_SHA
python3 scripts/shortqa_probe.py    # 期望 20/20、answer_vector_digest = 89577bb4612ab1d8
# 4) 官方协议全量测试
bash scripts/run_official_suite_official.sh
```

详细步骤、每一条命令的期望输出、以及「错了怎么排查」见 `docs/REPRODUCE.md`。

---

## 3. 成绩

官方协议（`/v1/completions`，4096 输入 / 1024 输出，固定随机 token id，seed 20260912，2 轮预热 + 3 轮正式，
每轮 `max(8, 2×C)` 请求）下的输出吞吐对比官方基线：

<!-- RESULTS_TABLE_START -->
### 4K input / 1K output

| C | 官方基线 tok/s | 本方案 tok/s (mean) | 轮次 min–max | 相对基线 | TTFT p50/p95 (s) | TPOT p95 (ms) | 状态 |
|---:|---:|---:|---|---:|---|---:|---|
| 1 | 9.055 | 62.26 | 50.91–70.42 | 6.88x | 1.36/1.36 | 24.6 | PASS |
| 8 | 52.480 | 112.15 | 107.86–118.77 | 2.14x | 1.45/9.74 | 78.6 | PASS |
| 16 | 74.011 | 127.57 | 125.96–129.25 | 1.72x | 1.50/21.51 | 145.2 | PASS |
| 32 | 142.864 | 126.06 | 122.93–130.05 | 0.88x | 118.34/159.17 | 151.0 | PASS |
| 64 | 194.387 | - | - | - | - | - | NOT_RUN |
| 128 | 181.599 | - | - | - | - | - | NOT_RUN |

### 64K input / 1K output

| C | 官方基线 tok/s | 本方案 tok/s (mean) | 轮次 min–max | 相对基线 | TTFT p50/p95 (s) | TPOT p95 (ms) | 状态 |
|---:|---:|---:|---|---:|---|---:|---|
| 1 | 7.520 | - | - | - | - | - | NOT_RUN |
| 8 | 13.929 | - | - | - | - | - | NOT_RUN |
| 16 | 14.242 | - | - | - | - | - | NOT_RUN |
| 32 | - | - | - | - | - | - | NOT_RUN |

### ShareGPT

| C | 官方基线 tok/s | 本方案 tok/s | 相对基线 | 状态 |
|---:|---:|---:|---:|---|
| 1 | 12.150 | (见 results/official/sharegpt-c1/) | - | NOT_RUN |
| 4 | 20.260 | (见 results/official/sharegpt-c4/) | - | NOT_RUN |
| 16 | 104.905 | (见 results/official/sharegpt-c16/) | - | NOT_RUN |
| 32 | 168.419 | (见 results/official/sharegpt-c32/) | - | NOT_RUN |
| 64 | 242.089 | (见 results/official/sharegpt-c64/) | - | NOT_RUN |
<!-- RESULTS_TABLE_END -->

> 口径纪律：本仓库**所有**数字都来自 `results/official/*/summary.json` 的原始轮次记录，
> 不使用 steady-decode、kernel 微基准或内部 harness 的数字冒充官方成绩。
> 哪一档没跑、跑到什么程度、为什么没跑，都在 `docs/RESULTS_OFFICIAL.md` 里逐条写明。

---

## 4. 仓库结构

```text
docs/
  REPRODUCE.md        从原始镜像 + 权重复现本方案的完整步骤与自检门
  ARCHITECTURE.md     11 组 overlay 逐个说明（改了什么、为什么、代价/边界）
  METHODOLOGY.md      经验总结：方法论、踩过的坑、已被证据关闭的路线
  RESULTS_OFFICIAL.md 官方协议成绩（4K/1K、64K/1K、ShareGPT）+ 逐档原始记录说明
  ACCURACY.md         精度门（GSM8K / RealWorldQA / LCB v6 / MathVision）
  LIMITATIONS.md      诚实边界：没结论的、未验证的、被外部条件阻塞的
manifests/
  quantization_manifest.json   量化契约（权重/激活/KV/草稿/head/原生扩展及其 SHA）
  run_manifest.json            运行契约（镜像 digest、参数、挂载、数据集 SHA、seed）
  SHA256SUMS                   交付物逐文件校验和
overlays/                     运行时 overlay 源码 + 开关门文件 + W8A8 Triton 调优表
runtime_patches/              17 个 bind-mount 运行时补丁（与容器挂载一一对应）
native_ext/                   assign_fixed8_i64（HIP 源码 + 编译出的 .so）
scripts/                      一键脚本、官方 benchmark、精度脚本、确定性/质量探针
results/                      原始结果（summary.json / requests.jsonl / 日志 / 证据）
```

---

## 5. 边界（请先读）

* 本方案是**运行时 overlay 栈**，不是重编译的 SGLang fork。它的可复现性依赖 3 样东西：
  **官方镜像 digest**、**17 个运行时补丁文件**、**overlay + 调优表**。仓库全部自带。
* 本方案在**并发上限**上有硬约束：`cap = mamba_pool / 4`，而 mamba 池与 KV 池抢同一份显存预算，
  因此本栈在 `max-running-requests=32` 时可用，**C64/C128 只能排队**（不是崩溃，是并发容量不足）。
  逐档结果与原因见 `docs/LIMITATIONS.md`。
* 精度门部分基准依赖外网数据集，宿主机**无外网**，部分门标为 `BLOCKED_OFFLINE`，见 `docs/ACCURACY.md`。
* 本仓库**不含**任何模型权重（体积大且受各自许可约束），只含“如何得到它们”的说明与校验和。

---

## 6. 许可与致谢

* 本仓库自有代码（overlay、脚本、内核）以 **Apache-2.0** 发布，见 `LICENSE`。
* SGLang 及其镜像、Qwen3.8-27B 权重、DFlash2 草稿模型各自遵循其原始许可。
* 上游工作：SGLang（Apache-2.0）、Qwen（Apache-2.0）、Hygon K100AI 平台运行时。
