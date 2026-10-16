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
### 4.1 冻结候选（DFLASH 投机 ON）—— **不通过**

| 门 | 分数 | 最低准入 | 判定 |
|---|---:|---:|---|
| GSM8K（n=414 中途读数，全量在跑） | **10.4%** | 95.0 | ❌ 不通过 |
| RealWorldQA / LCB v6 / MathVision | 见 `manifests/accuracy_results.json` | — | 受外网/沙箱限制，见 §2 |

### 4.2 对照：同一权重、关掉投机 —— **通过量级**

| 配置 | GSM8K |
|---|---:|
| DFLASH ON（链根 overlay + 调优表） | 12.5%（n=8） |
| **DFLASH OFF**（链根 overlay + 调优表） | **93.3%（n=150）**（中途读数曾 100%/n=10、96.7%/n=60） |

⇒ 精度失败的根因是**投机路径**，不是权重也不是 overlay 臂。
但关投机也只是 93.3%，**仍低于 95.0** ⇒ 投机是主因但不是唯一缺口。
完整受控实验与证据见 **[`ACCURACY_INVESTIGATION.md`](ACCURACY_INVESTIGATION.md)**。

见 `manifests/accuracy_results.json` 与 `results/official/acc_*/summary.json`。
<!-- ACC_RESULTS -->

---

## 5. 与"性能"的关系

* 本方案**不改变模型权重**（量化副本除外），overlay 中只有 FASTHEAD / CANDHEAD / SPLITKV / ATTNCFG
  属于"可能改变数值"的臂，它们的等价性证据见 `docs/ARCHITECTURE.md` 与 `docs/METHODOLOGY.md` §5。
* 精度门在**最终候选**上跑，配置与性能测试**完全一致**（同一端口、同一 `sweep.env`、同一权重）。
* 换权重副本、换量化格式、换投机配置，四项门必须重跑。
