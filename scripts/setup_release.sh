#!/usr/bin/env bash
# =============================================================================
#  setup_release.sh — 校验交付物并把它铺到 overlay 期望的绝对路径上
# =============================================================================
#  overlay 源码里硬编码了
#     _EXP=/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013
#  本脚本把仓库里的 overlays/ 复制到该路径（上游源码一行不改），
#  并把"门文件"按冻结内容写好、把探针门删干净。
#
#  用法：
#    bash scripts/setup_release.sh                 # 默认路径
#    EXP=/some/other/dir bash scripts/setup_release.sh
#    VERIFY_WEIGHTS=1 bash scripts/setup_release.sh   # 顺带校验权重 sha256
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${EXP:-/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013}"
VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-0}"
W8A8="${W8A8:-/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8}"
DRAFT="${DRAFT:-/data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2}"

say(){ printf '%s\n' "[setup] $*"; }
die(){ printf '%s\n' "[setup][FATAL] $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. 交付物校验
say "== 1. verify artifact SHA256 (manifests/SHA256SUMS) =="
[ -f "$REPO/manifests/SHA256SUMS" ] || die "manifests/SHA256SUMS missing"
( cd "$REPO" && sha256sum -c --quiet manifests/SHA256SUMS ) \
  || die "artifact SHA256 mismatch -> do not launch"
say "   all artifacts OK"

# ---------------------------------------------------------------- 2. 铺 overlay
say "== 2. install overlay tree -> $EXP =="
mkdir -p "$EXP"
for d in "$REPO"/overlays/*/; do
  name="$(basename "$d")"
  [ "$name" = "gates" ] && continue
  [ "$name" = "tables" ] && { mkdir -p "$EXP/tables"; cp -a "$d". "$EXP/tables/"; continue; }
  mkdir -p "$EXP/$name"
  cp -a "$d". "$EXP/$name/"
  say "   installed $name"
done

# ---------------------------------------------------------------- 3. 门文件
say "== 3. write frozen gate files =="
G="$REPO/overlays/gates"
cp -a "$G/candhead_cfg.txt"     "$EXP/candhead_cfg.txt"
cp -a "$G/combo2_cfg.txt"       "$EXP/combo2_cfg.txt"
cp -a "$G/gdn_fuse_cfg.txt"     "$EXP/gdn_fuse_cfg.txt"
cp -a "$G/prefill_gdn_cfg.txt"  "$EXP/prefill_gdn_cfg.txt"
cp -a "$G/q8b8_cfg.txt"         "$EXP/q8b8_cfg.txt"
cp -a "$G/attn_cfg.txt"         "$EXP/attn_cfg.txt"
cp -a "$G/head64_sids.txt"      "$EXP/head64_sids.txt"

# SPLITKV / TAILSYNC 是"存在即开"的臂，内容必须精确
printf '16\n' > "$EXP/draft_splitkv_cfg.txt"
: > "$EXP/tail_sync_cfg.txt"

[ "$(cat "$EXP/draft_splitkv_cfg.txt")" = "16" ] || die "draft_splitkv_cfg.txt != 16"
[ ! -s "$EXP/tail_sync_cfg.txt" ]                || die "tail_sync_cfg.txt must be empty"
for f in candhead_cfg.txt combo2_cfg.txt gdn_fuse_cfg.txt prefill_gdn_cfg.txt q8b8_cfg.txt; do
  [ "$(cat "$EXP/$f")" = "1" ] || die "$f != 1"
done
[ "$(cat "$EXP/attn_cfg.txt" | tr -s ' ')" = "16 64 8" ] || die "attn_cfg.txt != '16 64 8'"
say "   gates OK"

# --------------------------------------------------- 3b. 并发精度修复必须已铺好
#  没有 concurrency_fix_overlay 时，bs>=3 的 target verify 会掉回 parent varlen，
#  GSM8K 全量并发 8 会从 ~96% 崩到 8.49%（已实测）。
say "== 3b. concurrency accuracy fix present =="
[ -f "$EXP/concurrency_fix_overlay/sitecustomize.py" ] \
  || die "concurrency_fix_overlay/sitecustomize.py missing -> bs>=3 accuracy will collapse"
if grep -q "1 <= batch <= _BATCH_MAX" "$EXP/concurrency_fix_overlay/sitecustomize.py"; then
  say "   q8 verify widening present (batch 1..32)"
else
  die "concurrency_fix_overlay missing the q8 verify widening"
fi
if grep -q "exact_get_last_loc" "$EXP/concurrency_fix_overlay/sitecustomize.py"; then
  say "   exact get_last_loc present"
else
  die "concurrency_fix_overlay missing exact get_last_loc"
fi

# ---------------------------------------------------------------- 4. 探针门必须不存在
say "== 4. ensure no probe gates ship enabled =="
for f in chainprobe_cfg.txt specprobe_cfg.txt accgate_cfg.txt; do
  if [ -e "$EXP/$f" ]; then say "   removing probe gate $f"; rm -f "$EXP/$f"; else say "   probe gate absent $f"; fi
done

# ---------------------------------------------------------------- 5. 权重与草稿
say "== 5. weights present? =="
[ -f "$W8A8/config.json" ]      || die "target W8A8 dir not found: $W8A8"
[ -f "$DRAFT/model.safetensors" ] || die "draft model not found: $DRAFT"
say "   target: $W8A8  ($(ls "$W8A8"/model-*-of-*.safetensors 2>/dev/null | wc -l) shards)"
say "   draft : $DRAFT"
if [ "$VERIFY_WEIGHTS" = "1" ]; then
  say "== 5b. verify target weight SHA256 (slow, ~30 s) =="
  ( cd "$W8A8" && sha256sum -c "$REPO/manifests/target_weights_sha256.txt" ) || die "weight SHA mismatch"
  say "   weights OK"
fi

# ---------------------------------------------------------------- 6. 离线前置
say "== 6. offline prerequisites =="
mkdir -p "$REPO/tvmffi_cache"
if [ ! -f "$REPO/tvmffi_cache/libtorch_c_dlpack_addon_torch210-rocm.so" ]; then
  say "   WARNING: tvmffi_cache/ is empty."
  say "            On a host WITHOUT internet the tvm_ffi first-import build stalls forever."
  say "            Copy the built .so from a host that has one, or build it once with network."
fi

say ""
say "DONE. Now run:  bash scripts/launch_official_lane.sh"
say "Expected boot geometry: max_total_num_tokens=166720, max_running_requests=16, avail mem ~10.27 GB"
