set -euo pipefail
# 冻结配置来源（按优先级）：
#   1) Q38_SWEEP_CFG 显式指定
#   2) $Q38_EXP/sweep_official.env（launch_official_lane.sh 会传 Q38_EXP，
#      setup_release.sh 会把仓库内 sweep_official.env 铺到该位置）
#   3) 宿主历史路径（仅作为老机器回退）
CFG="${Q38_SWEEP_CFG:-${Q38_EXP:+$Q38_EXP/sweep_official.env}}"
CFG="${CFG:-/data/qwen38-27b-k100ai-int8-opt/contest_submission_20261017/sweep_official.env}"
PORT="${SWEEP_PORT:-18636}"
NCCL_PORT="${SWEEP_NCCL_PORT:-29636}"
SWEEP_SPEC_ALGO=DFLASH
SWEEP_GRAPH_ARGS="--cuda-graph-max-bs 16"
SWEEP_MAMBA_TRACK=256
SWEEP_DRAFT_TOKENS=8
SWEEP_NUM_STEPS=1
SWEEP_MAX_REQS=32
SWEEP_MEM_FRAC=0.90
SWEEP_PIECEWISE=off
SWEEP_KV_DTYPE=bfloat16
SWEEP_CHUNK=16384
SWEEP_EXTRA_ARGS="--max-mamba-cache-size 64"
SWEEP_PROFILE_TIMES=0
SWEEP_W8A8_METHODS=1
SWEEP_TRITON_JSON_DIR=""
SWEEP_LMSLIM_TUNING_JSON=""
SWEEP_HIP_VISIBLE="0,1"
SWEEP_EXTRA_PYTHONPATH=""
# 链首必须是 concurrency_fix_overlay：它先跑冻结的上游链，再装
# (1) 全模块 exact get_last_loc 与 (2) q8 verify 放宽到 batch<=32。
# 少了它，bs>=3 的 target verify 全部掉回 parent varlen 路径：
# C8 吞吐 178 -> ~110，GSM8K 并发精度 ~96% -> ~8.5%。
SWEEP_SITECUSTOMIZE_PREFIX="/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013/concurrency_fix_overlay"
SWEEP_EXTRA_ENV=""
SWEEP_DRAFT_QUANT=""
if [[ -f "$CFG" ]]; then set +u; set -a; source "$CFG"; set +a; set -u; fi
# 卡号以 launch_official_lane.sh 的 GLOBAL_CARDS 为准（Q38_HIP_VISIBLE 优先），
# 避免"设备映射用 A 组卡、HIP_VISIBLE 用 B 组卡"的两套事实来源。
if [[ -n "${Q38_HIP_VISIBLE:-}" ]]; then SWEEP_HIP_VISIBLE="$Q38_HIP_VISIBLE"; fi
echo "[SWEEP] algo=$SWEEP_SPEC_ALGO graph='$SWEEP_GRAPH_ARGS' track=$SWEEP_MAMBA_TRACK draft=$SWEEP_DRAFT_TOKENS steps=$SWEEP_NUM_STEPS reqs=$SWEEP_MAX_REQS mem=$SWEEP_MEM_FRAC pw=$SWEEP_PIECEWISE kv=$SWEEP_KV_DTYPE chunk=$SWEEP_CHUNK"
export SGLANG_DFLASH2_PROFILE_TIMES="$SWEEP_PROFILE_TIMES"
export W8A8_SUPPORT_METHODS="$SWEEP_W8A8_METHODS"
if [[ -n "$SWEEP_TRITON_JSON_DIR" ]]; then export TRITON_JSON_DIR="$SWEEP_TRITON_JSON_DIR"; fi
if [[ -n "$SWEEP_LMSLIM_TUNING_JSON" ]]; then export LMSLIM_TUNING_JSON="$SWEEP_LMSLIM_TUNING_JSON"; fi
export HIP_VISIBLE_DEVICES="$SWEEP_HIP_VISIBLE"
if [[ -n "$SWEEP_EXTRA_PYTHONPATH" ]]; then
  export PYTHONPATH="$SWEEP_EXTRA_PYTHONPATH:${PYTHONPATH:-}"
fi
if [[ -n "$SWEEP_SITECUSTOMIZE_PREFIX" ]]; then
  export SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX="$SWEEP_SITECUSTOMIZE_PREFIX"
fi
# optional extra env passthrough (KEY=VAL KEY=VAL ...)
if [[ -n "$SWEEP_EXTRA_ENV" ]]; then
  for kv in $SWEEP_EXTRA_ENV; do export "${kv?}"; done
fi
echo "[SWEEP] dflash2_profile_times=$SGLANG_DFLASH2_PROFILE_TIMES w8a8_methods=$W8A8_SUPPORT_METHODS triton_json_dir=${TRITON_JSON_DIR:-<default>}"
MODEL=/tmp/q38-target-model
VIDEO_DEPS=/tmp/q38-video-deps
SRC=/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8
rm -rf "$VIDEO_DEPS"
DECORD_WHEEL="${Q38_DECORD_WHEEL:-/data/qwen38-27b-k100ai-int8-opt/release/staging_official_tp2_safe_r2_20260918/payload/vendor/decord-0.6.0-py3-none-manylinux2010_x86_64.whl}"
if [[ -f "$DECORD_WHEEL" ]]; then
  PYTHONPATH= python3 -m pip install --disable-pip-version-check --no-index --no-deps --target "$VIDEO_DEPS" "$DECORD_WHEEL"
  export PYTHONPATH="$VIDEO_DEPS:${PYTHONPATH:-}"
else
  # 纯文本 lane 不需要 decord；缺 wheel 只影响视频输入。
  echo "[SWEEP][WARN] decord wheel not found: $DECORD_WHEEL (text lanes unaffected)" >&2
fi
rm -rf "$MODEL"; mkdir -p "$MODEL"
for f in "$SRC"/*; do ln -s "$f" "$MODEL/$(basename "$f")"; done
ln -sfn /data/my_models/Qwen/Qwen3.8-27B/preprocessor_config.json "$MODEL/preprocessor_config.json"
ln -sfn /data/my_models/Qwen/Qwen3.8-27B/video_preprocessor_config.json "$MODEL/video_preprocessor_config.json"
spec_args=()
if [[ "$SWEEP_SPEC_ALGO" == "NONE" ]]; then
  # Adjudication support (2026-10-16): run the lane with NO speculator at all.
  # Note spec_args is appended AFTER $SWEEP_EXTRA_ARGS, so this cannot be
  # overridden from the command line; it needs its own branch here.
  spec_args=()
elif [[ "$SWEEP_SPEC_ALGO" == "DFLASH" ]]; then
  if [[ -n "${SWEEP_DRAFT_QUANT:-}" ]]; then
    spec_args=(--speculative-algorithm DFLASH --speculative-draft-model-path "${SWEEP_DRAFT_MODEL_PATH:-/data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2}" --speculative-draft-model-quantization "$SWEEP_DRAFT_QUANT" --speculative-draft-attention-backend triton --speculative-num-steps "$SWEEP_NUM_STEPS" --speculative-num-draft-tokens "$SWEEP_DRAFT_TOKENS")
  else
    spec_args=(--speculative-algorithm DFLASH --speculative-draft-model-path "${SWEEP_DRAFT_MODEL_PATH:-/data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2}" --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton --speculative-num-steps "$SWEEP_NUM_STEPS" --speculative-num-draft-tokens "$SWEEP_DRAFT_TOKENS")
  fi
else
  spec_args=(--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens "$SWEEP_DRAFT_TOKENS")
fi
pw_args=()
if [[ "$SWEEP_PIECEWISE" == "off" ]]; then pw_args=(--disable-piecewise-cuda-graph); fi
exec python3 -u -m sglang.launch_server \
 --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --nccl-port "$NCCL_PORT" --served-model-name Q38-OFFICIAL-CONTEST-20261017 \
 --chat-template /data/qwen38-dflash2-k100ai/runtime_assets/qwen38_chat_template.jinja \
 --trust-remote-code --dtype bfloat16 --kv-cache-dtype "$SWEEP_KV_DTYPE" --tp-size 2 --pp-size 1 \
 --attention-backend fa3 --mm-attention-backend fa3 --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 --mamba-track-interval "$SWEEP_MAMBA_TRACK" \
 $SWEEP_GRAPH_ARGS "${pw_args[@]}" --page-size 64 --context-length 262144 --mem-fraction-static "$SWEEP_MEM_FRAC" \
 --chunked-prefill-size "$SWEEP_CHUNK" --max-prefill-tokens 16384 --max-running-requests "$SWEEP_MAX_REQS" \
 --custom-all-reduce-backend native --enable-metrics --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
 $SWEEP_EXTRA_ARGS \
 "${spec_args[@]}"
