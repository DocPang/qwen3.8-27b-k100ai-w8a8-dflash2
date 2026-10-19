# 独立复现验证（2026-10-19）

环境：2 × K100AI（gfx928，卡 4/5，TP2），镜像
`harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317d…c3aca68`，
仓库 `9bab2ff` 及之后，**运行时树全部来自仓库 `overlays/vendor/`**
（容器内 `/data/qwen38-27b-k100ai-int8-opt` 由 `$Q38_RUNTIME_ROOT` 挂载覆盖，
宿主原目录不可见）。

| 目录 | 内容 | 结果 |
|---|---|---|
| `indep-c8/` | 4K/1K C8，2 预热 + 3 正式，16 req/round | 171.30 tok/s（188.89 / 184.76 / 140.25），TTFT p50 1.88 s |
| `indep-64k-c1/` | 64K/1K C1，1 预热 + 1 正式，1 req/round | 19.44 tok/s，TTFT p50 29.2 s |
| `indep_shortqa.json` | 20 题质量门 | 20/20，digest `89577bb4612ab1d8` |

同栈的 C1 逐位确定性锁（`scripts/ab3.py rel`）：
`code` SHA `9351bc6049cfa597` / accept 3.4595，`records` SHA `ee000f28af8a215d` / accept 6.1687。

对照发布值：C8 177.87（-3.7%，轮间波动）、64K C1 19.20（+1.3%）。
