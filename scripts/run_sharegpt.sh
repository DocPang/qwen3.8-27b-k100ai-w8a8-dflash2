#!/usr/bin/env bash
set -u
ROOT=/data/qwen38-27b-k100ai-int8-opt/contest_20260923
OUTROOT="$ROOT/results/strongest-20260923"
DATA="$ROOT/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
CONTAINER=q38-contest-best-laout3072-hcu67-20260923
MODEL_PATH=/tmp/q38-target-model
SERVED=Q38-CONTEST-BEST-LAOUT3072-HCU67-20260923
BASE="${BASE:-http://127.0.0.1:18580}"

if [[ -f "$OUTROOT/fixed_matrix_runner.pid" ]]; then
  p=$(cat "$OUTROOT/fixed_matrix_runner.pid" || true)
  while [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; do sleep 10; done
fi

run_one() {
  local c=$1
  local out="$OUTROOT/sharegpt-c${c}"
  mkdir -p "$out"
  (hcu-snapshot || true) > "$out/hcu_snapshot_before.txt" 2>&1
  docker inspect "$CONTAINER" > "$out/container_inspect.json" 2>/dev/null || true
  sha256sum "$DATA" > "$out/dataset.sha256"
  for round in 1 2 3; do
    local file="$out/round-$(printf '%02d' "$round").jsonl"
    local log="$out/round-$(printf '%02d' "$round").stdout.log"
    if [[ -s "$file" ]]; then
      echo "SKIP_EXISTING sharegpt-c${c} round=$round"
      continue
    fi
    local warm=$((2*c))
    echo "START sharegpt-c${c} round=$round $(date -Is)"
    docker exec "$CONTAINER" python3 -u -m sglang.bench_serving       --backend sglang-oai       --base-url "$BASE"       --dataset-name sharegpt       --dataset-path "$DATA"       --model "$MODEL_PATH"       --served-model-name "$SERVED"       --sharegpt-context-len 131072       --num-prompts 1000       --max-concurrency "$c"       --request-rate inf       --warmup-requests "$warm"       --seed 20260912       --disable-tqdm       --output-details       --output-file "$file"       > "$log" 2> "$out/round-$(printf '%02d' "$round").stderr.log"
    echo $? > "$out/round-$(printf '%02d' "$round").exitcode"
    echo "END sharegpt-c${c} round=$round $(date -Is)"
  done
  (hcu-snapshot || true) > "$out/hcu_snapshot_after.txt" 2>&1
}

for c in 1 4 16 32 64; do
  run_one "$c"
done
