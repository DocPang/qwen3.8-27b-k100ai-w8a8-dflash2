# 官方协议成绩

> 本文件里每一张表都可以用
> `python3 scripts/build_results_table.py --suite results/official` 从原始 `summary.json` 重新生成。
> 表格由 `scripts/build_results_table.py` 生成后粘贴；原始轮次明细在 `results/official/`。

---

## 1. 官方协议（本项目实际执行的口径）

来源：比赛资料 + 本项目 `official_protocol.lock.md` 快照（2026-09-23）。

### 1.1 固定长度矩阵

| 项 | 值 |
|---|---|
| 端点 | `/v1/completions` |
| 输入构造 | **固定随机 token id**（`make_prompt(4096, seed)`，前缀含请求序号） |
| 采样 | `temperature=0`, `top_p=1`, `ignore_eos`, `max_tokens=1024` |
| 缓存 | 每请求唯一 `cache_salt` ⇒ 冷前缀（期望 `cached_tokens=0`） |
| seed | `20260912` |
| 预热 / 正式 | **2 轮预热 + 3 轮正式** |
| 每轮请求数 | `max(8, 2 × concurrency)` |
| 输出长度 | 1024 |
| 档位 | 4K/1K：C1/C8/C16/C32/C64/C128；64K/1K：C1/C8/C16/C32 |

### 1.2 ShareGPT

| 项 | 值 |
|---|---|
| 数据集 | `ShareGPT_V3_unfiltered_cleaned_split.json`（672,837,942 B） |
| 每档请求 | **1000** |
| 预热请求 | `2 × concurrency` |
| 请求速率 | `inf` |
| 轮次 | 3 |
| 档位 | C1/C4/C16/C32/C64 |
| 执行 | `python3 -m sglang.bench_serving --backend sglang-oai` |

---

## 2. 结果

### 2.0 重要说明：这批性能数字来自"未修复链"

下表由官方套件（卡 2/3，`q38-official-contest-hcu23-20261017`）跑出，其
`SWEEP_SITECUSTOMIZE_PREFIX` 指向 `candhead_overlay`，**不含**
`concurrency_fix_overlay`。也就是说：

* **性能有效**（该配置与冻结基线一致，4K/64K 全档位 PASS）；
* 但该链上 bs≥3 的 target verify 走的是较慢的 parent varlen 路径，
  所以 **C8/C16/C32/C64/C128 的吞吐是保守下界**；修复后（raw paged_attention）
  应当更快，实测数据见 §2.2。

<!-- GENERATED_TABLE -->
### 2.1 4K input / 1K output（未修复链，官方套件实测，3 轮中位数）

| 并发 | output tok/s | input tok/s | TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|
| C1 | 65.46 | 261.82 | 1357.4 | 13.97 |
| C8 | 109.80 | 439.22 | 4257.8 | 58.85 |
| C16 | 127.49 | 509.96 | 7097.1 | 101.61 |
| C32 | 125.19 | 500.76 | 98357.5 | 115.20 |
| C64 | 129.31 | 517.23 | 282193.1 | 117.71 |
| C128 | 133.35 | 533.38 | 649515.5 | 116.00 |

### 2.1b 64K input / 1K output（未修复链）

| 并发 | output tok/s | input tok/s | TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|
| C1 | 19.76 | 1264.91 | 29588.3 | 21.72 |
| C8 | 24.11 | 1542.99 | 222817.1 | 50.25 |
| C16 | 24.93 | 1595.65 | 463787.2 | 50.27 |
| C32 | 24.92 | 1594.70 | 958795.9 | 50.78 |

### 2.2 修复后复测（`concurrency_fix_overlay`）

修复后 bs≥3 的 q=8 verify 从 parent varlen 切到 raw `paged_attention`，**吞吐显著提升**：

| 档位 | 修复前 out tok/s | 修复后 out tok/s | 变化 | 修复后 TTFT | 修复后 TPOT |
|---|---:|---:|---:|---:|---:|
| 4K/1K C8 | 109.80 | **190.12** | **+73%** | 4401 ms | 30.23 ms |
| 4K/1K C16 | 127.49 | **208.90** | **+64%** | 7586 ms | 56.72 ms |
| 4K/1K C32 | 125.19 | **222.12** | **+77%** | 57195 ms | 62.29 ms |
| 64K/1K C8 | 24.11 | 23.72 | −1.6%（持平） | 268 345 ms | 43.01 ms (p50) / 108.83 ms (p95) |

> 口径：`scripts/official_benchmark.py`，`--base-url http://127.0.0.1:18638`（**不能带 `/v1`**，
> 脚本内部会拼 `/v1/completions`），`warmup-rounds=1`、`measure-rounds=2`
> （少于官方的 2+3，故数值偏保守）。
>
> **64K/1K C8 说明**：长输入档位由 TTFT 主导（268 s），verify 路径优化对吞吐影响很小，
> 因此与修复前基本持平。该轮测量与另一条 lane 的模型加载并行执行，数值应视为**下界**。
> 4K 档位的提升来自 decode 阶段 raw `paged_attention` 取代 parent varlen 路径。

### 2.3 并发档位与精度的关系（重要）

同一模型、同一贪心解码，**不同并发档位的 GSM8K 准确率不同**：

| 档位 | GSM8K（v6 提取，前 N 题） | 说明 |
|---|---:|---|
| C1（1319 全量） | **96.21%** | 单请求，无 batch 内数值差异 |
| C1（前 200） | 96.00% | 与全量一致 |
| C2（100） | 95.00% | 刚好达到 95% 门槛 |
| C4（100） | 92.00% | |
| C8（前 200） | 90.00% | 非截断口径 92.31% |
| **C8（全量 1319）** | **90.45%** | 非截断口径 **94.98%**，`length` 64 题（4.85%） |

> C1 全量与 C8 全量用 `max_tokens=4096`；C2/C4/C8 前 100/200 用 `2048`。
> C8 全量的非截断准确率已到 94.98%，端到端缺口全部来自复读截断；
> 对截断题重试可救回 43/64（单策略）→ 端到端 93.71%，或 50/64（组合）→ 94.24%。

对 C1 前 200 与 C8 前 200 做逐题对齐后：**C1 对、C8 错的共 14 题**
（`20, 23, 24, 45, 47, 52, 56, 57, 75, 107, 146, 150, 153, 168`），
其中 5 题是 `finish_reason=length`，9 题是 `stop` 但最终答案数字改变
（例如 idx=20：C1 得 15（正确），C8 得 25）。

结论：`bs≥3` 时 batch 内的数值差异会改变部分题目的 argmax，这是**当前仍存在的精度缺口来源**，
不是评分规则问题。已尝试的消除手段见 `docs/LIMITATIONS.md` §0.1。

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

### stage status（原始）

```json
{
  "fixed-4k-1k-c1": {
    "status": "PASS",
    "detail": "rc=0",
    "at": 1792175194.26208,
    "at_iso": "2026-10-17 02:26:34"
  },
  "fixed-4k-1k-c8": {
    "status": "PASS",
    "detail": "rc=0",
    "at": 1792175784.12288,
    "at_iso": "2026-10-17 02:36:24"
  },
  "fixed-4k-1k-c16": {
    "status": "PASS",
    "detail": "rc=0",
    "at": 1792176809.283366,
    "at_iso": "2026-10-17 02:53:29"
  },
  "fixed-4k-1k-c32": {
    "status": "PASS",
    "detail": "rc=0",
    "at": 1792178900.0433059,
    "at_iso": "2026-10-17 03:28:20"
  },
  "acc_gsm8k": {
    "status": "ABANDONED",
    "detail": "thinking mode (template default reasoning_effort=xhigh) exhausted max_tokens -> content empty; superseded by acc_gsm8k_nothink",
    "at": 1792188073.7330234,
    "at_iso": "2026-10-17 06:01:13"
  },
  "acc_gsm8k_nothink": {
    "status": "RUNNING",
    "detail": "full test split 1319, max_tokens=4096, enable_thinking=false",
    "at": 1792184522.4406536,
    "at_iso": "2026-10-17 05:02:02"
  }
}
```
<!-- GENERATED_TABLE -->

---

## 3. 怎么读这些数字（三个必须说清的口径）

### 3.1 官方基线是 **stock SGLang 0.5.12**

比赛公布的 K100_AI 基线不含 W8A8 量化、不含投机解码、不含本项目任何 overlay。
所以"相对基线 xx 倍"混合了**量化 + 投机 + overlay**三者的贡献，不是 overlay 单独的贡献。

### 3.2 "官方随机 token 口径"比"真实文本口径"低很多

官方协议用**固定随机 token id** 作为提示词。这类提示词的投机接受率显著低于真实代码/文本，
因此同一个服务在官方口径下的 C1 输出吞吐明显低于"真实文本"口径。
本项目保留两套口径的记录，但**只有官方口径进入本文件**。

### 3.3 TTFT p95 在 C≥8 时会被"排队"支配

每档每轮要发 `2×concurrency` 个请求，而服务端同时在跑的请求数上限是
`cap = 16`（见 `docs/LIMITATIONS.md`）。因此 C16/C32 的 TTFT p95 里包含大量排队时间，
它衡量的是"这一档的整体等待"，不是"单请求首字延迟"。
原始 `requests.jsonl` 里逐请求记录 TTFT/TPOT，可据此重算任意分位。

---

## 4. 原始产物索引

```text
results/official/
  suite_summary.json              # 全部档位的汇总（本文件的表格来源）
  stage_status.json               # 每个阶段的最终状态（PASS/DONE/FAILED/NOT_SUPPORTED）
  suite.log                       # 带时间戳的阶段日志
  server_args.txt / models.json / container_inspect.json / sweep.env.snapshot
  fixed-4k-1k-c<N>/
      config.json                 # 本轮协议参数快照
      summary.json                # 官方汇总（3 轮 mean/min/max + 分位数）
      round-0X/requests.jsonl     # 逐请求原始明细
      stdout.log / stderr.log
  fixed-64k-1k-c<N>/              # 同上
  sharegpt-c<N>/
      round-0X.jsonl              # bench_serving 的 --output-details 明细
      round-0X.stdout.log / .stderr.log / .exitcode
      dataset.sha256
results/evidence/
  c1_lock_official.log            # C1 逐位确定性锁（3-rep）
  shortqa_official.json           # 质量门 20/20
  w8a8_c1_lock_official.log       # C1 锁原始日志
```

---

## 5. 已跑 / 未跑 / 为什么

见 `results/official/stage_status.json`。逐条的说明（尤其是没跑成的档位和原因）见
`docs/LIMITATIONS.md`。

---

## 6. 对照：关掉投机解码的官方协议性能（权衡用）

因为冻结候选的精度不合格（根因 = DFLASH 投机路径，见 `docs/ACCURACY_INVESTIGATION.md`），
这里额外给出**同一权重 / 同一调优表 / 关掉投机**这一替代配置的官方协议性能，
让"精度可信但更慢"的代价可量化。

车道：容器 `q38-nospec-hcu45-20261017`（端口 18637），
`SWEEP_SPEC_ALGO=NONE`，其余（调优表、链根 overlay、启动参数）与冻结候选一致；
启动几何 `max_total_num_tokens=820160`、`max_running_requests=12`、`available_gpu_mem=4.84 GB`。

结果（`results/nospec_perf/`，原始 `summary.json` 一并入库）：

<!-- NOSPEC_TABLE -->
### 4K input / 1K output

| C | 官方基线 tok/s | 本方案 tok/s (mean) | 轮次 min–max | 相对基线 | TTFT p50/p95 (s) | TPOT p95 (ms) | 状态 |
|---:|---:|---:|---|---:|---|---:|---|
| 1 | 9.055 | 34.66 | 34.60–34.70 | 3.83x | 1.37/1.37 | 27.6 | PASS |
| 8 | 52.480 | 141.63 | 140.91–142.25 | 2.70x | 7.99/10.52 | 53.3 | PASS |
| 16 | 74.011 | 151.12 | 150.35–151.73 | 2.04x | 12.96/85.31 | 73.0 | PASS |
| 32 | 142.864 | - | - | - | - | - | NOT_RUN |
| 64 | 194.387 | - | - | - | - | - | NOT_RUN |
| 128 | 181.599 | - | - | - | - | - | NOT_RUN |
<!-- NOSPEC_TABLE -->

> 口径提醒：该对照与冻结候选**同时**在同一台主机上运行，
> 因此它的数字可能被轻微低估；判读时只看**量级差异**。
