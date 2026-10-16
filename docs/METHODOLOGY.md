# 经验总结：这台机器上什么有效、什么无效、以及为什么

本文是本方案的"经验"部分。它不重复参数（见 `ARCHITECTURE.md`），而是回答：
**在一个真实的 DCU 双卡栈上做性能优化，哪些做法被证据支持、哪些被证据杀掉、哪些坑必须提前知道。**

---

## 1. 方法论：五条不可让步的纪律

### 1.1 外部量尺优先，内部指标只能解释因果

比赛官方协议（固定 4K/1K、64K/1K、ShareGPT 三套矩阵 + 四项精度门）被当作**唯一外部量尺**。
steady-decode t/s、kernel 微基准、单层 graph 延迟一律只作为**解释变量**，不作为成绩，也不作为晋级依据。
原因是实测反复证明它们会误导：

* 同一个"纯 decode 100 t/s"在不同 prompt/输出长度下差异巨大（`out_len=512 → accept 8.00 → 117 t/s`，
  `out_len=384 → accept 6.975 → 108 t/s`），所以"100 t/s"必须钉死 (prompt, out_len, cache 状态, ignore_eos)。
* 带同步点的阶段计时会把 host 等待算进 GPU 时间；capture 期的 profiler 表会混入编译活动。

### 1.2 匹配 A/B：同一台机器、同一对卡、同一规则版本、同一数据集/seed

跨机器、跨镜像世代、跨权重的结果一律不比较；世代间必须放"锚点"（见 §4.4）。

### 1.3 每一条改动都要过两道门

| 门 | 内容 | 不通过的处置 |
|---|---|---|
| **C1 逐位锁** | 固定 `code` / `records` 两个真实文本负载，3-rep，要求 `TEXT_SHA` + `ACCEPT_MED` **逐位复现**，且 `cached_tokens=0` | 直接 REJECT，回滚 |
| **质量门** | 20 题 plain 问答，要求 `correct=20/20`、`empty=0`、`answer_vector_digest` 等于参考值 | 直接 REJECT |

这两道门抓到了很多"看起来更快"的改动（见 §5 表格）。

### 1.4 预注册判据，事后不改

每个实验在跑之前写死：证伪条件、判定阈值、以及"如果结果落在中间带怎么办"。
一个反面教材：早期曾用两点拟合外推第三个宽度点，结果偏差 27.7%（阈值 15%），
导致"固定项 27.4 ms = 内存地板"这个**后来必须撤回**的结论。
**教训：禁止用两点外推做预算算术。**

### 1.5 fail-closed + 启动后断言

所有 overlay 都是 fail-closed（缺门文件 ⇒ 静默 inert、不报错）。
这在实验期是优点（不会把环境搞崩），在**发布期是陷阱**：
一个缺失的门文件会让"一键复现"产出一个**与实测不同的栈**，而且没有任何报错。
因此发布脚本必须：(a) 校验每个交付物 SHA256；(b) 缺失的门文件按冻结内容创建并校验内容；
(c) 启动后逐臂 grep banner，缺任一臂即 ABORT。

---

## 2. 这台机器上的物理事实（带实测数字）

这些结论决定了后面所有杠杆的取舍，复现者可以直接借用。

1. **整步不是带宽受限，而是"内核数量 × 每核成本"受限。**
   以 `DFlashVerifyInput.verify`（每步恰好一次）为锚精确计步：**≈2499 个内核/步**；
   窗口设备时间合计 ≈57.9 ms/步 ≈ 整步的 84% ⇒ 几乎没有重叠，
   `68.6 ms ÷ 2499 ≈ 27.4 µs/内核`。因此**减少发射次数**（融合）比"省字节"更有账可算。
   （同一批数据也否证了另一个极端：权重流地板只有 ≈22.5 ms，整步是它的 3.05×。）

2. **并发上限由 mamba 池决定，而不是 KV 池。**
   `cap = mamba_pool // 4`；mamba 池与 KV 池抢同一份显存预算。
   `cap16`（pool 64）时 `max_total_num_tokens=166720`、`available_gpu_mem≈10.3 GB`。
   实测 KV 使用率只有 0.30，**KV 从不成为约束**。

3. **prefill 与 decode 严格串行**（`scheduler.py` "Run prefill first"；DFLASH 强制
   `disable_overlap_schedule` / `enable_mixed_chunk=False`）。
   墙钟 ≈ prefill 时间 + decode 时间。

4. **`TTFT ≈ Σ(prefill_tokens) / prefill_throughput`**，且 prefill 吞吐**与批大小基本无关**
   （实测 3125 / 4209 / 3217 / 3190 tok/s @ C1/C2/C3/C4）。
   直接测量：3 个 4382-token 请求同时到达时 TTFT = **4.09 s**（≈13146 token ⇒ 3217 tok/s），
   与观测到的 4.3 s"入场量子"几乎完全吻合 ⇒ 那天梯就是 prefill chunk 的完成时间。
   推论：**把请求打包得更细/更粗都不改变吞吐**，"chunk 与 max-prefill 调优"因此关闭。

5. **真实步长与 harness 读出的 step_ms 差 28–31%。**
   用 `SGLANG_RECORD_STEP_TIME` 拿真实逐步墙钟：`bs=12` 真步长 **128.3 ms**，
   而 harness 的 `step_ms` 是 167.8 ms。且**52.3% 的 forward 墙钟落在 prefill 长间隙里**。
   ⇒ 拿 harness 的 step_ms 做微观账要非常小心。

6. **投机接受率随 batch 单调下降且可复现。**
   `bs=1` 时 accept ≈ 6.9，`bs≥8` 时 ≈ 2.87。这是本栈**最大也最难攻**的剩余杠杆
   （若 C16 能把 accept 从 2.87 恢复到 bs=1 的水平，墙钟 66 s → ~38 s）。
   已定位：分歧由 DFLASH 投机路径引入（底座无投机时 C8 与 C1 输出**逐位相同**），
   而且在 `compute_candidates` 的**输入**就已经分歧 ⇒ 指向 draft 前向/其输入的分块边界行相关归约。

7. **`page_size=64` 不是可调项**：`--mamba-scheduler-strategy no_buffer` 会直接报
   "Paged KV cache block size must be 64"。

8. **内存几何本身会影响 C1 锁**：`max_total_num_tokens` 224000 ⇒ 破锁；166720 ⇒ 稳；138112 ⇒ 破锁。
   机制未闭合，因此发布时**必须**把几何写进校验清单（本仓库的期望几何见 `REPRODUCE.md` §5.2）。

---

## 3. 真正兑现了的杠杆（按贡献排序）

| # | 杠杆 | 兑现的收益 | 证据强度 |
|---|---|---|---|
| 1 | cap16 波形（mem 0.90 / mamba pool 64 / graph 16）+ 补齐 pinned 解覆盖缺口 | C32 **134.2 t/s（+6.4%）**、C32 TTFT **32.57 s（−21.2%）**、C16 126.5 | 同 nonce A/B/A + C1 锁 + 质量门 |
| 2 | CANDHEAD（草稿候选头钉解） | C16 **117.1→135.5（+15.7%）**、C32 109.8→121.7（+10.8%）、C8 step 137→113.9 ms | 5-rep C1 锁 + 质量门 |
| 3 | FASTHEAD 全档位钉解 | 高并发单步省 12–32 ms | 480/480 argmax 一致 |
| 4 | W8A8 Triton 调优表 | 单步省 ~5.2 ms；官方四并发配对 +10.4%~+21.1% | 全桶 `SPLIT_K=1`（逐位） |
| 5 | raw-q8 verifier 扩到 bs≤8（Q8B8） | C8 并发下 16 层全注意力不再落慢路径 | 门限经 **bs≤12 反证**后确定 |
| 6 | SPLITKV 16 路 | 单步省 ~1.9 ms | `NUM_SPLITS=1` 先做零风险 smoke |
| 7 | TAILSYNC | 去掉每步一次 D2H 同步（最高 6.25 ms/次量级） | 首次调用仍走厂商路径 |
| 8 | GDNFUSE + PREFILL_GDN（COMBO2） | 砍 144 次/步冗余发射；prefill 稳定性 | 纯数据搬运逐位验证 / 纯加法补丁 |
| 9 | `assign_fixed8_i64_c16exp_v2` | 槽位映射内核 batch 上限 8→16 | 64-trial 离线门 + 运行时 AUDIT |
| 10 | ATTNCFG decode tile 16×64×8 | 草稿 decode attention 并行度 | prefill 构造性不受影响 |

作为量级参考：这一族改动把**单流真实文本**的 decode 从 66–74 t/s 提到 **103–105 t/s（+41%~+58%）**，
端到端 4K/1K **75.2 → ≈102 t/s（+34%）**，逐步硬件耗时 **72.1 → 57.9 ms（−19.6%）**。

---

## 4. 仪器与判据纪律（这一节的教训最贵）

### 4.1 accept 混淆：并发 A/B 只报聚合吞吐一定被带偏

`agg_e2e ≈ accept × requests × 1000 / step_ms`。accept 的 run-to-run 带宽是 2.28–4.41，
足以吃掉任何 ±5% 的改动。**判据必须是"同时报 accept 与 accept-free 代理（`agg_e2e/accept`）"。**
一个真实教训：早前一次 A/B 报出 `+14.7%`，事后按 accept-free 复核发现是 accept 抽签造成的。

### 4.2 噪声带不能用"两个 off 臂的均值差"来估

同一次实验里 `A` 与 `Ap` 只差 0.05%，据此判某臂"超带"；同 nonce 复测时两个 off 臂差 1.6%、
同臂两 rep 差 3.5%。
**更正后的规则**：(1) 基线取两个 off 臂的**合并均值**；(2) 噪声带取**臂内 spread 的最大值**；
(3) 判"真实增益"要求同时**超带**且**在目标口径上同向兑现**；(4) 重要臂用 **≥3 rep 或交织测量**。
这条规则**追溯适用于所有用"A vs Ap 均值差"或单点 rep 当带的旧结论**。

### 4.3 探针/仪器本身会污染测量

三例：
* 关闭 CUDA Graph 做诊断时，target_verify 从 50.6 ms 涨到 189 ms ⇒ **图确实在 replay**，
  "启动开销是差距来源"这条假设被自己的诊断反证。
* `fwdprobe` 的异步计时曾给出"12288 token 打包 prefill 只要 273 ms（≈45000 tok/s）"，
  按 27B 模型算这是 2.4 PFLOP/s，物理不可能 ⇒ 该探针作废，且**以它为前提的改造方案一并作废**。
* 并发期做 A/B 时，兄弟容器的活动污染了两次"破锁"判定 ⇒ 两次 REJECT 被**撤销**。
  **纪律：破锁结论必须在独占条件下复现。**

### 4.4 "target 输出 SHA 相同"不能证明草稿路径等价

换草稿注意力后端（fa3）时，草稿算出垃圾 token（accept 4.5714 → 1.0711、吞吐 62 → 14.8 t/s），
而 **target 输出 SHA 完全不变**。⇒ 任何"输出一致即等价"的推理对草稿路径都不成立。

### 4.5 跨世代必须放锚点

同一镜像、同一参数在不同世代可能因为 dtype/编译缓存差异而漂移。
做法：每个世代都用同一批参照请求记录 `TEXT_SHA` + `accept_length`；对不上就说明有跨世代漂移，
当次结果只能用于**计时**，不能用于**内容**比较。

---

## 5. 已被证据关闭的路线（禁区表）

这些不是"没试"，是"试了、有数据、关闭了"。列出来是为了别重复踩。

| # | 禁区 | 关闭依据 |
|---|---|---|
| 1 | W8A8/INT8 草稿量化 | 装载成功但 accept 1.01、输出长度 7.55× ⇒ REJECT |
| 2 | "追 accept 本身"当作独立优化项 | accept 崩塌由**投机路径**引入、且不在我方 overlay/表内 |
| 3 | 只看 kernel 设备时间 / 发射数量来估收益 | L5 删 239 次发射 **0 收益**（设备时间占比口径失效） |
| 4 | raw-q8 verifier 放宽到 `batch ≤ 12` | accept-free step 全级退化（C8 +12.1%、C16 +6.3%、C32 +8.3%）⇒ `bs≤8` 才正确 |
| 5 | chunk 32768 / max-prefill 粒度 | 32768 更慢；prefill 吞吐与 chunk 尺寸无关（±3%） |
| 6 | overlap schedule / mixed chunk | DFLASH spec v1 硬禁；实测调度间隙仅 0.7–1.0 ms |
| 7 | `--mamba-scheduler-strategy no_buffer` | 直接报错：block size 必须 64 |
| 8 | `mem_frac 0.90 + mamba pool 32`；cap14 / cap20 / cap24 | 破 C1 锁或 C32 崩（cap20/cap24 双 REJECT） |
| 9 | 按 accept 重调 head pinned-solution 表 | ±7% 且方向不一致 |
| 10 | draft 长度 8→4 / 自适应 draft | accept −30~45%、步长不变、几何破锁 |
| 11 | `--enable-deterministic-inference` | 启动即失败（triton 持久化 matmul 需要 98304 B smem > 65536 B 硬件上限） |
| 12 | `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true` | **3-rep 下破 C1 锁**（1-rep 曾假绿）⇒ REJECT |
| 13 | `--triton-attention-reduce-in-fp32` | C1 锁保持但无稳定增益、C8 波动 ⇒ 不晋级 |
| 14 | 替换 `compute_candidates` 的 BF16 matmul 为 batch 不变实现 | 分歧在其**输入**（同 prompt 跨请求 `in_diff` 已达 1.36→5.67）⇒ 无效 |
| 15 | 怀疑 `assign_fixed8_i64_c16exp_v2.so` 导致锁漂移 | 64 trial / bs=1..16 全 PASS + 运行时 AUDIT ⇒ 免责 |
| 16 | "cap12 在 C16/C32 上比 cap16 快 10–14%" | 同 nonce A/B/A 证伪：C32 反而 cap16 +5.4% 且 TTFT 好 23%；C16 只 +2.7% |
| 17 | 底座批量解码的 batch 不变性 | 无投机时 C8 与 C1 输出**逐位相同** ⇒ 分歧确实在投机路径 |
| 18 | 整模型 `torch.compile` | 20 分钟未编译完 + FA pybind graph break ⇒ REJECT_AS_CONFIGURED（**注意：这没证明"编译无帮助"**） |
| 19 | 草稿分组卷积单独编译 | 图内实测上限 **0.97%** 整周期 ⇒ 连 5% 门槛的 1/5 都不到 |
| 20 | 模型级 W8A8→W4A8 降字节 | 潜在 ~20%，但改变模型数值 ⇒ 需用户拍板的模型级变更，不在本方案内 |

---

## 6. 未闭合的最大问题

1. **accept 的 batch 依赖**（§2.6）——唯一的大杠杆，需要引擎级项目来解，
   而不是配置层实验。
2. **C1 锁对内存几何的非单调性**（224000 破 / 166720 稳 / 138112 破）——机制未解释。
3. **一个未复现的服务卡死**：真实长上下文工具会话（~42k token、25 tools）曾触发
   `chunk_gated_delta_rule → prepare_chunk_indices` 处 300 s 看门狗自杀；
   同镜像受控 42k 前缀探针正常 ⇒ 触发条件 UNKNOWN，需要重复实验定性。

---

## 7. 给复现者的建议（按性价比排序）

1. **先跑通两道门，再谈性能。** C1 逐位锁 + 20/20 质量门是这套栈的全部信任基础。
2. **把几何写进检查清单。** `max_total_num_tokens / max_running_requests / graph bs / cap` 四个数必须逐字对上。
3. **每次只动一个变量，并且同 nonce 匹配。** 想看并发，必须同时报 accept 与 accept-free 代理。
4. **遇到"更快但输出变了"，先当 bug 处理。** 本仓库里所有非逐位的臂（FASTHEAD/CANDHEAD/SPLITKV/ATTNCFG）
   都有各自的等价性证据（argmax 不变、top-16 不变、content sha + accept 对照）。
5. **改并发容量要从 mamba 池入手**，并且准备好接受 "cap 增加 ⇒ 破锁" 的风险。
