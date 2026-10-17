# DFlash 乘性惩罚修复（可选，**默认不启用**）

这个 overlay 让 `repetition_penalty` 第一次真正作用到 DFlash 投机验证的 logits 上。
**它在技术上是成功的，但在 GSM8K 上是净负效果，因此不进入交付配置。**

## 修的是什么

上游有两处缺陷，叠加后让所有惩罚参数在投机路径上完全失效：

1. `speculative/dflash_info.py` 的 `verify` 里，`sampling_info.apply_logits_bias()` 被作用在
   一个**全零缓冲** `linear_penalty` 上。加性项（`logit_bias` / presence / frequency /
   `vocab_mask`）能穿过零缓冲存活，但乘性 `apply_scaling_penalties` 对 0 恒等
   （`where(0<0, 0*s, 0/s) = 0`），所以 `repetition_penalty` 无论设多少都不生效。
2. `managers/schedule_batch.py` 在投机模式下提前 `return`，跳过 `cumulate_output_tokens`。
   EAGLE 在 `eagle_worker.py` 里补了这一步，DFlash 全仓没有。于是
   `cumulated_repetition_penalties` 恒为 ones、`cumulated_frequency/presence` 恒为 zeros。

## 为什么走 `req.output_ids` 而不是 orchestrator

试过两条 orchestrator 路线，都失败：

* 包装 `DFlashWorker._prepare_for_speculative_decoding` 调 `cumulate_output_tokens`：
  该函数内部有 `if not self._is_prepared: return`，而 batch 级 prepare 发生得更晚，
  所以**静默跳过**（实测 `cumulated` 计数恒为 0）。
* 在 verify 里调 `orch.prepare()` 再累积：`BatchedPenalizerOrchestrator` 没有 `prepare`
  方法（`AttributeError`），且 `DFlashVerifyInput` 每次新建导致跨 step 的 `prev_bonus` 丢失。

最终实现绕开 orchestrator，**直接从 `batch.reqs[i].output_ids[-512:]` 构造惩罚**：

```python
sub = rows.index_select(1, idx)          # rows = logits[i*r:(i+1)*r]
rows[:, idx] = torch.where(sub < 0, sub * rp, sub / rp).to(rows.dtype)
```

只改历史 token 的那几列（每步几万元素），而不是分配 `bs x vocab` 的 32 MB 张量。

## 实测结果（C8，前 300 题，`max_tokens=4096`）

| 配置 | `length` | 端到端 |
|---|---:|---:|
| 基线（`rp=1.0`） | 12 | 92.667% |
| 本 overlay + `rp=1.05` | 11 | 93.000% |
| 本 overlay + `rp=1.10` | **14** | **89.667%** |

日志可以证明惩罚确实生效：`[PR78] mulpen applied rp=1.100 hist=N rows=8`。
但 `rp=1.05` 只多 1 题（在正负 3% 的题间波动内），`rp=1.10` 反而更差且截断变多——
过度惩罚破坏了正常推理链。

## 结论

复读的根因是**思考阶段失控**（`length` 题 `reasoning_content` 中位数 15351 字符，
正常题只有 401 字符），不是采样参数缺失。修好乘性惩罚只证明了"缺陷确实存在"，
并不能把复读题转成正确题。**交付配置保持 `repetition_penalty=1.0`（不加载本 overlay）。**

如果要复现实验，把 `SWEEP_SITECUSTOMIZE_PREFIX` 指向本目录，并设
`Q38_DFLASH_MULPEN=1`，请求里带 `repetition_penalty>1.0`。
