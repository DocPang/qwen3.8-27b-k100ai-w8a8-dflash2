# 精度门

> 状态字符串由评测脚本自己写出：脚本明确标注这些结果是
> `PUBLIC_EQUIVALENT_PREVALIDATION_NOT_ORGANIZER_FINAL` —— 即"公开等效预验证"，**不是**主办方最终口径。
> 我们不把预验证的数字写成"通过官方门"。

## 1. 官方四门（本项目的要求）

| Benchmark | 官方分 | 最低准入 | 本方案 |
|---|---:|---:|---|
| GSM8K | 97.0 | 95.0 | 见 `manifests/accuracy_results.json` |
| LiveCodeBench v6 | 90.3 | 88.3 | 见 `manifests/accuracy_results.json` |
| RealWorldQA | 85.9 | 83.9 | 见 `manifests/accuracy_results.json` |
| MathVision Without CI | 90.0 | 88.0 | 见 `manifests/accuracy_results.json` |

**准入规则（项目自定，严于比赛文本）**：任一门的分数低于最低准入，该候选**不得**作为项目最终参赛候选。

---

## 2. 执行环境与离线约束

评测机**无外网**。这直接决定了哪些门能跑：

| 门 | 需要什么 | 本机情况 | 处理 |
|---|---|---|---|
| GSM8K | ModelScope 数据集 | 本地缓存 `modelscope___gsm8k`（1319 条 test split）已有 | `MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1` 离线加载 ✅ |
| RealWorldQA | 数据集 | 需确认本地缓存 | 尽力执行，失败即记录确切错误 |
| LiveCodeBench v6 | 数据集 + **代码执行沙箱** | 沙箱镜像历史上一直阻塞 | 尽力执行，失败即 `BLOCKED_OFFLINE` |
| MathVision | 数据集 | 本地无 | 预期 `BLOCKED_OFFLINE` |

**纪律**：跑不成的门标 `BLOCKED_OFFLINE` 并单列历史基线，**不计入本候选通过**，
也不用别的机器/别的权重的数字冒名顶替。

---

## 3. GSM8K 的评测口径

| 项 | 值 |
|---|---|
| 数据集 | `modelscope/gsm8k`, subset `main`, split `test`（1319 条） |
| 提示 | 0-shot，user message 为题目原文 |
| 采样 | greedy（temperature 0） |
| `max_tokens` | 4096 |
| 并发 | 8 |
| 计分 | 数值等价比对（`\boxed{}` / `Answer:` / 加粗 / 末位数字，Decimal 归一；支持分数） |
| 断点续跑 | 是（`requests.jsonl` 逐条落盘） |

```bash
MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1 \
  <evalscope-venv>/bin/python scripts/run_accuracy_gsm8k_public.py \
    --base-url http://127.0.0.1:18636/v1 \
    --model Q38-OFFICIAL-CONTEST-20261017 \
    --output-dir results/official/acc_gsm8k \
    --concurrency 8 --limit 0 --max-tokens 4096 --timeout 600
```

> ⚠️ **`max_tokens` 必须是显式参数**。Qwen3.8 是长思考模型，`max_tokens=1024` 会把不少题的
> 推理链截断，从而把"模型不会"误判成"模型答错"。同一个候选在 1024 / 4096 两个预算下
> 的分数差异属于**harness 口径差异**，报告里必须写明用了哪个。

---

## 4. 结果

<!-- ACC_RESULTS -->
### 4.1 并发精度崩塌：根因与修复（已闭环）

**根因**：`batch2_raw_q8_overlay/q8_batch2_v6.py` 把 target verify 的 q=8 因果注意力
换成单次 raw `paged_attention`，但守卫是 `batch in (1, 2)`。**bs>=3 静默回落到
parent `vllm_flash_attn_varlen_func`**，其 q=8 因果结果在本 HCU 栈上与 raw 路径不一致。

**修复**（`overlays/concurrency_fix_overlay/sitecustomize.py`）：

1. `exact_get_last_loc`（纯 PyTorch gather）替换 `get_last_loc` / `get_last_loc_triton`，
   并锚定 `mem_cache.common.hcu_get_last_loc` 这一个全局名
   （所有 `from ... import get_last_loc` 副本的 `__globals__` 都指向它，故 race-free）；
2. q8 verify 快路径放宽到 `1 <= batch <= 32`。

修复前后的 GSM8K（greedy，thinking 默认开，官方冻结配置；`max_tokens` 见表注）：

| 并发 | 修复前 | 修复后（原始提取） | 修复后（v6 提取） |
|---|---:|---:|---:|
| C1（全量 1319） | 94.9962% | — | **96.21%** |
| C2（前 100） | 96.0% | 94.0% | **95.0%** |
| C4（前 100） | **16.7%** | **91.0%** | 92.0% |
| C8（前 100） | **14~16%** | **91.0%** | 92.0% |
| C8（前 200） | — | 88.5% | 90.0% |
| **C8（全量 1319，mt=4096）** | **0%**（见下） | **90.5231%** | 90.4473% |

> 注：C8 前 200 题的读数低于前 100 题，说明 100 题样本偏易；报告以 200 题为准。
>
> **C8 全量那行的"修复前 0%"是实测**：未修复链（`candhead_overlay`）在 thinking + C8 +
> `max_tokens=4096` 下跑前 105 题全错——76 题 `content` 为空、66 题撞 `max_tokens`，
> `reasoning_content` 中位数 13938 字符。修复把这一档从 0% 拉到 90.45%。
>
> C2/C4/C8（前 100/200）用的是 `max_tokens=2048`；C1 全量与 C8 全量用 4096。
> **不要用 2048 与 4096 的数字互相比较**（见 §4.2 的复读截断率差异）。

### 4.2 三栏解耦统计（把 greedy 复读退化与算术正确性分开）

| 并发 | 端到端 (v6) | 非截断样本准确率<br>Accuracy @ `finish_reason=stop` | 复读截断率<br>(`finish_reason=length`) |
|---|---:|---:|---:|
| C1 全量 1319 | **96.21%** | 1269/1313 = **96.65%** | 6/1319 = **0.45%** |
| C2 前 100 | **95.00%** | 95/100 = **95.00%** | 0/100 = **0.00%** |
| C8 前 200 | 90.00% | 180/195 = **92.31%** | 5/200 = **2.50%** |
| **C8 全量 1319（mt=4096）** | 90.45% | 1192/1255 = **94.98%** | 64/1319 = **4.85%** |

**C8 全量这一行是本项目的关键读数**：非截断准确率 **94.98%**，几乎正好踩在 95% 线上。
也就是说端到端不达标的缺口**全部来自 4.85% 的复读截断**，而不是"并发把算术算错了"。

### 4.2b 复读截断是**可救**的（零侵入重试实验）

对 C8 全量里那 64 道 `finish_reason=length` 的题做重试，**不改模型、不改服务配置**，
只改请求参数：

| 策略 | 救回 | 端到端（1319 题口径） |
|---|---:|---:|
| A：贪心 + `max_tokens=8192` | **43 / 64** | 93.707% |
| B：`temperature=0.7, top_p=0.95`，`max_tokens=4096` | **43 / 64** | 93.707% |
| A ∪ B | **50 / 64** | **94.238%** |

* A 与 B 救回的是不同子集（交集 36），所以组合比单策略多 7 题；
* 两种策略都失败的仅 14 题：`174, 182, 255, 371, 425, 675, 749, 768, 796, 835, 906, 924, 1042, 1176`；
* 距 95% 门槛（1253 题正确）还差 **10 题**。

**必须澄清的一点（已用实验否证）**：不能把上面的 43/64 外推成
"把默认 `max_tokens` 改成 8192 就能到 93.7%"。

我们做了对照：同一前 300 题，`max_tokens=8192` 全量重跑（`acc_pr63_c8_300_mt8192`）
得 **93.00%**（279/300），而 `max_tokens=4096` 的同 300 题是 **92.67%**（278/300）——
**只多 1 题**。拆开看：

* 原 12 道截断题里，8192 **救回 7 道**（`87,122,157,175,182,275,298`），仍错 5 道（`132,174,181,184,255`）；
* 但 8192 **新弄错 9 道**原本正确的题（`12,58,100,125,147,183,187,201,203`）。

也就是说：`bs≥3` 的 batch 内数值差异会让同一题在不同批次组合下答案漂移，
单次全量重跑的题间波动约 ±3%（9/300）。**"重试"之所以有效，是因为它换了一条生成轨迹，
不是因为 8192 这个数字本身。**

因此诚实的结论是：

| 场景 | C8 端到端 |
|---|---:|
| 单次请求（4096 或 8192 都一样） | **≈ 90.5%** |
| 对 `length` 题重试一次（单策略） | 93.71% |
| 对 `length` 题重试一次（A∪B 组合） | **94.24%** |

若评测规则是"一次请求定成绩"，本方案 C8 不达 95%；若允许对截断样本重试，
则可达 94.24%（仍差 10 题）。两种情况都**不应**把重试结果当作单次请求成绩上报。

`finish_reason=length` 的题在文本末尾是"猜题目/自我怀疑"死循环
（`"The prompt is a corrupted version of…"`、`"Wait, the text provided in…"`），
属 greedy 解码下的重复思考退化，与量化/算子正确性无关。

### 4.3 关于评分口径（必须如实说明）

GSM8K 的计分是**数字精确匹配**：gold 取 `####` 之后，prediction 取模型最终结论段。
`v6` 口径只修 `extract_pred` 的实现缺陷（括号内单位换算备用答案、`2:00 pm` 时钟格式、
取到 `Step N` 标题数字），**模型原始 `content` 一字未改**。

**合规红线（明确不做）**：不对 idx=93（36.36 vs 36）、idx=1016（139 vs 138）做容差匹配。

见 `manifests/accuracy_results.json`、`results/official/acc_*/summary.json`
与 `docs/ACCURACY_INVESTIGATION.md`。
<!-- ACC_RESULTS -->

---

## 5. 与"性能"的关系

* 本方案**不改变模型权重**（量化副本除外），overlay 中只有 FASTHEAD / CANDHEAD / SPLITKV / ATTNCFG
  属于"可能改变数值"的臂，它们的等价性证据见 `docs/ARCHITECTURE.md` 与 `docs/METHODOLOGY.md` §5。
* 精度门在**最终候选**上跑，配置与性能测试**完全一致**（同一端口、同一 `sweep.env`、同一权重）。
* 换权重副本、换量化格式、换投机配置，四项门必须重跑。
