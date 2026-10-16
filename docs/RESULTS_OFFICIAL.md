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

<!-- GENERATED_TABLE -->
（由 `scripts/build_results_table.py` 生成；见 `results/official/suite_summary.json`）
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
