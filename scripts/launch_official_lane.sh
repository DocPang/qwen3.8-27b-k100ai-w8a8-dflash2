#!/usr/bin/env bash
# =============================================================================
#  launch_official_lane.sh — 按冻结挂载表/环境起容器（本方案实测的那一个）
# =============================================================================
#  前置：bash scripts/setup_release.sh 已跑过（overlay 已铺到 $EXP、权重已就位）
#  用法：
#    R132=renderD130 R131=renderD131 bash scripts/launch_official_lane.sh
#  说明：HIP_VISIBLE_DEVICES 用 **全局卡号**，本脚本默认 2,3（对应 renderD130/131）。
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMG="${IMG:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}"
CT="${CT:-q38-official-contest-hcu23-20261017}"
PORT="${PORT:-18636}"
# 默认沿用 18636 -> 29636 的偏移；换 PORT 并行起第二条 lane 时不会撞 nccl 端口。
NCCL_PORT="${NCCL_PORT:-$((PORT + 11000))}"
R132="${R132:-renderD130}"
R131="${R131:-renderD131}"
GLOBAL_CARDS="${GLOBAL_CARDS:-2,3}"
EXP="${EXP:-/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013}"

# tvm_ffi 首次 import 会现场编译 torch 扩展；离线机器上会无限期卡住。
# 仓库不携带预编译产物，所以这里 fail-closed：要么放好 .so，要么显式放行。
if [ ! -f "$REPO/tvmffi_cache/libtorch_c_dlpack_addon_torch210-rocm.so" ] \
   && [ "${Q38_ALLOW_TVMFI_BUILD:-0}" != "1" ]; then
  echo "[launch][FATAL] tvmffi_cache/libtorch_c_dlpack_addon_torch210-rocm.so missing." >&2
  echo "                离线机器会卡在 tvm_ffi 首次编译（/health 长 000）。" >&2
  echo "                放好预编译 .so，或设 Q38_ALLOW_TVMFI_BUILD=1 让容器联网自建。" >&2
  exit 1
fi

SP="sglang/srt"
SPK="sglang/kernels/ops"
PY="python3.10/dist-packages"
BASE_MOUNTS=""

add_mount(){ BASE_MOUNTS="${BASE_MOUNTS} -v $(printf '%q' "$1"):$(printf '%q' "$2")"; }

# ---- 16 个运行时补丁（源路径与仓库内 runtime_patches/ 树一一对应） ----
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/runtime_patch_official_tp2_native_hcu_policy_20260914/custom_all_reduce.py" \
          "/usr/local/lib/$PY/$SP/distributed/device_communicators/custom_all_reduce.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/runtime_patch_spec_gap2_markers_v1_20260926/$SP/speculative/eagle_worker_v2.py" \
          "/usr/local/lib/$PY/$SP/speculative/eagle_worker_v2.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/runtime_patch_newbase_r4e_nb023_20260913/$SP/models/qwen3_5_mtp.py" \
          "/usr/local/lib/$PY/$SP/models/qwen3_5_mtp.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/runtime_patch_newbase_r4e_nb023_20260913/$SP/managers/mm_utils.py" \
          "/usr/local/lib/$PY/$SP/managers/mm_utils.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/release/staging_official_tp2_safe_r2_20260918/payload/runtime_patch_core_r1_qwen38_toolparser_v1/$SP/function_call/qwen3_coder_detector.py" \
          "/usr/local/lib/$PY/$SP/function_call/qwen3_coder_detector.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/release/staging_official_tp2_safe_r2_20260918/payload/runtime_patch_r2_mm_config_v1/$SP/multimodal/processors/base_processor.py" \
          "/usr/local/lib/$PY/$SP/multimodal/processors/base_processor.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/release/staging_official_tp2_safe_r2_20260918/assets/qwen38_chat_template.jinja" \
          "/data/qwen38-dflash2-k100ai/runtime_assets/qwen38_chat_template.jinja"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/runtime_patch_official_tp2_gateup_triton_v5d_cache_20260915/W8A8_17408_5120_gfx928_120cu.json" \
          "/usr/local/lib/$PY/lightop/configs/w8a8/W8A8_17408_5120_gfx928_120cu.json"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/experiments/dflash_hip_target_sampling_20261012/dflash_worker_sparse_q.py" \
          "/usr/local/lib/$PY/$SP/speculative/dflash_worker.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/experiments/dflash_hip_target_sampling_20261012/dflash_hip_target_only.py" \
          "/usr/local/lib/$PY/$SP/speculative/dflash_hip_target_only.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/experiments/dflash_hip_target_sampling_20261012/dflash_utils.py" \
          "/usr/local/lib/$PY/$SP/speculative/dflash_utils.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/experiments/dflash_hip_target_sampling_20261012/dflash_selector_rejection.py" \
          "/usr/local/lib/$PY/$SP/speculative/dflash_selector_rejection.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/experiments/dflash_hip_target_sampling_20261012/dflash_info_sparse_q.py" \
          "/usr/local/lib/$PY/$SP/speculative/dflash_info.py"
add_mount "$REPO/runtime_patches/data/qwen38-27b-k100ai-int8-opt/experiments/spec_repair_pagedkv_20261008/overlay/extend_attention.py" \
          "/usr/local/lib/$PY/$SP/layers/attention/triton_ops/extend_attention.py"
add_mount "$REPO/runtime_patches/data/qwen38-dflash2-k100ai/work/sourcefind_sglang_overlay_idletrim_v1/python/$SP/models/dflash.py" \
          "/usr/local/lib/$PY/$SP/models/dflash.py"
add_mount "$REPO/runtime_patches/data/qwen38-dflash2-k100ai/work/sourcefind_sglang_overlay_idletrim_v1/python/$SPK/speculative" \
          "/usr/local/lib/$PY/$SPK/speculative"

echo "[launch] image  = $IMG"
echo "[launch] name   = $CT   port=$PORT   cards=$GLOBAL_CARDS ($R132,$R131)"
echo "[launch] overlay root = $EXP"

docker rm -f "$CT" >/dev/null 2>&1 || true

eval docker run -d --name "$CT" \
  --privileged --ipc host --network host --entrypoint bash \
  --device /dev/kfd --device /dev/dri/$R132 --device /dev/dri/$R131 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
  -e SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX="$EXP/concurrency_fix_overlay" \
  -e PYTHONPATH="$EXP/concurrency_fix_overlay" \
  -e Q38_EXP="$EXP" \
  -e Q38_HIP_VISIBLE="$GLOBAL_CARDS" \
  -e SWEEP_PORT="$PORT" \
  -e SWEEP_NCCL_PORT="$NCCL_PORT" \
  -e TRITON_JSON_DIR="$EXP/tables" \
  -e SGLANG_Q38_ASSIGN_FIXED8_I64_SO="$REPO/native_ext/k100_assign_fixed8_i64_c16exp_v2.so" \
  -e Q38_INTEGRATION_V7=1 -e Q38_ENABLE_SPEC=1 -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_Q38_NATIVE_CAR_HCU_POLICY=1 -e SGLANG_USE_LIGHTOP=1 \
  -e SGLANG_USE_CUDA_IPC_TRANSPORT=1 -e ALLREDUCE_STREAM_WITH_COMPUTE=1 \
  -e SGLANG_Q38_COMPACT_HEAD_M1=0 -e SGLANG_Q38_TP2_COMPACT_HEAD_M1=0 \
  -e Q38_SPEC_FULLHEAD_M1_SOLUTION=10347 \
  -e Q38_MTP_GATEUP_M4_SOLUTION=10009 -e Q38_MTP_FULLHEAD_M4_SOLUTION=11055 \
  -e HIP_PATH=/opt/dtk/hip -e ROCM_PATH=/opt/dtk \
  -e SGLANG_DFLASH2_PROFILE_TIMES=0 -e SGLANG_DFLASH2_STAGE_SYNC=0 \
  -e SGLANG_DFLASH2_DEBUG_STATS=0 -e SGLANG_DFLASH2_DEBUG_DIAG_ROUNDS=0 \
  -v /data:/data -v /opt/hyhal:/opt/hyhal \
  -v "$REPO/tvmffi_cache":/root/.cache/tvm-ffi \
  -v "$REPO/scripts/entry_official_contest.sh":/probe/candidate_entry.sh \
  $BASE_MOUNTS \
  "$IMG" /probe/candidate_entry.sh

echo "[launch] waiting for /health (up to 10 min) ..."
for i in $(seq 1 60); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://127.0.0.1:$PORT/health" || true)
  if [ "$c" = "200" ]; then echo "[launch] HEALTHY after $((i*10))s"; break; fi
  sleep 10
done
[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true)" = "200" ] \
  || { echo "[launch][FATAL] not healthy - see: docker logs $CT"; exit 1; }

echo "[launch] == boot geometry =="
docker logs "$CT" 2>&1 | grep -E "max_total_num_tokens" | tail -1
echo "[launch] == arm banner assertion =="
docker logs "$CT" > /tmp/launch_boot.log 2>&1
MISS=0
for tag in "Q38 CANDHEAD" "Q38 COMBO2" "Q38 GDNFUSE" "Q38 HCU01 FASTHEAD" "Q38 PREFILL" "Q38 SPLITKV" "Q38 TAILSYNC"; do
  if grep -qF "[$tag]" /tmp/launch_boot.log; then echo "   arm live  $tag"; else echo "   arm MISSING $tag"; MISS=1; fi
done
[ "$MISS" = 0 ] || { echo "[launch][FATAL] frozen stack incomplete -> ABORT"; exit 1; }

echo "[launch] DONE. service: http://127.0.0.1:$PORT  (model Q38-OFFICIAL-CONTEST-20261017)"
