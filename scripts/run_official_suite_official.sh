#!/usr/bin/env bash
# Official contest full suite — Qwen3.8-27B W8A8 INT8 + DFlash2 champion
# lane: q38-w8a8tune-hcu23-20261013 (HCU2/3, port 18634)
# Resumable: a stage already recorded as PASS/DONE is skipped.
set -u
ROOT=/data/qwen38-27b-k100ai-int8-opt
BENCH=$ROOT/contest_20260923/official/benchmark.py
ACC=$ROOT/contest_20260924_accuracy
SUITE=$ROOT/contest_submission_20261017/official_suite_official
SG_DATA=$ROOT/contest_20260923/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
BASE=http://127.0.0.1:18636
MODEL=Q38-OFFICIAL-CONTEST-20261017
CONTAINER=q38-official-contest-hcu23-20261017
mkdir -p "$SUITE"
STATUS=$SUITE/stage_status.json
LOG=$SUITE/suite.log

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
mark() { python3 - "$STATUS" "$1" "$2" "$3" <<'PY'
import json,os,sys,time
p,stage,status,detail=sys.argv[1:5]
d=json.load(open(p)) if os.path.exists(p) else {}
d[stage]={"status":status,"detail":detail,"at":time.time(),"at_iso":time.strftime("%F %T")}
json.dump(d,open(p,"w"),indent=2,ensure_ascii=False)
PY
}
get_status() { python3 -c "
import json,os
p='$STATUS'
if not os.path.exists(p): print('NONE'); raise SystemExit
print(json.load(open(p)).get('$1',{}).get('status','NONE'))
"; }
wait_health() {
  for _ in $(seq 1 120); do
    c=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" || true)
    [ "$c" = "200" ] && return 0
    sleep 10
  done
  return 1
}
snapshot_server() {
  local out=$1
  docker inspect "$CONTAINER" > "$out/container_inspect.json" 2>/dev/null || true
  docker exec "$CONTAINER" bash -lc 'ps -ef | grep -m1 -o "sglang.launch_server.*"' > "$out/server_args.txt" 2>/dev/null || true
  curl -s "$BASE/v1/models" > "$out/models.json" 2>/dev/null || true
  cp "$ROOT/experiments/hcu01_sweep_20261013/sweep.env" "$out/sweep.env.snapshot" 2>/dev/null || true
}
run_perf() {   # $1=label  $2=input_len  $3=concurrency
  local label=$1 ilen=$2 conc=$3
  local out=$SUITE/$label
  [ "$(get_status "$label")" = "PASS" ] && { say "SKIP_PASS $label"; return 0; }
  local req=$((2*conc)); (( req < 8 )) && req=8
  mkdir -p "$out"; snapshot_server "$out"
  mark "$label" RUNNING "input=$ilen conc=$conc"
  say "START $label input=$ilen conc=$conc requests=$req"
  python3 -u "$BENCH" --base-url "$BASE" --model "$MODEL" --output-dir "$out" \
    --input-len "$ilen" --output-len 1024 --concurrency "$conc" \
    --warmup-rounds 2 --measure-rounds 3 --requests-per-round "$req" \
    --seed 20260912 --timeout 3600 > "$out/stdout.log" 2> "$out/stderr.log"
  local rc=$?
  local st=$(python3 -c "import json;print(json.load(open('$out/summary.json')).get('status'))" 2>/dev/null || echo NO_SUMMARY)
  mark "$label" "$st" "rc=$rc"
  say "END $label status=$st rc=$rc"
}
run_perf_set() { local ilen=$1; shift; for c in "$@"; do run_perf "fixed-$((ilen/1024))k-1k-c${c}" "$ilen" "$c"; done; }
run_acc_gsm8k() {
  [ "$(get_status acc_gsm8k)" = "PASS" ] && { say "SKIP_PASS acc_gsm8k"; return 0; }
  local out=$SUITE/acc_gsm8k; mkdir -p "$out"
  mark acc_gsm8k RUNNING "full test split 1319 max_tokens=4096"
  say "START acc_gsm8k"
  MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1 \
  "$ACC/evalscope-venv/bin/python" -u "$ACC/scripts/run_accuracy_gsm8k_public.py" \
     --base-url "$BASE/v1" --model "$MODEL" --output-dir "$out" \
     --concurrency 8 --limit 0 --max-tokens 4096 --timeout 600 \
     > "$out/stdout.log" 2> "$out/stderr.log"
  local rc=$?
  local acc=$(python3 -c "import json;print(json.load(open('$out/summary.json'))['accuracy_pct'])" 2>/dev/null || echo NA)
  mark acc_gsm8k "$([ "$rc" = 0 ] && echo DONE || echo FAILED)" "rc=$rc accuracy_pct=$acc"
  say "END acc_gsm8k rc=$rc accuracy=$acc"
}
run_acc_evalscope() {   # $1=stage $2=datasets $3=limit $4=extra
  local stage=$1 ds=$2 lim=$3 extra=${4:-}
  [ "$(get_status "$stage")" = "PASS" ] && { say "SKIP_PASS $stage"; return 0; }
  local out=$SUITE/$stage; mkdir -p "$out"
  mark "$stage" RUNNING "datasets=$ds limit=$lim"
  say "START $stage datasets=$ds limit=$lim extra=$extra"
  MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1 \
  "$ACC/evalscope-venv/bin/evalscope" eval \
     --model "$MODEL" --model-id "$MODEL" --eval-type openai_api \
     --api-url "$BASE/v1/chat/completions" --datasets "$ds" --limit "$lim" \
     --work-dir "$out" --no-timestamp --ignore-errors $extra \
     > "$out/stdout.log" 2> "$out/stderr.log"
  local rc=$?
  mark "$stage" "$([ "$rc" = 0 ] && echo DONE || echo FAILED)" "rc=$rc"
  say "END $stage rc=$rc"
}
run_sharegpt() {
  local c=$1 stage="sharegpt-c${c}" out=$SUITE/$stage
  [ "$(get_status "$stage")" = "PASS" ] && { say "SKIP_PASS $stage"; return 0; }
  mkdir -p "$out"
  sha256sum "$SG_DATA" > "$out/dataset.sha256"
  mark "$stage" RUNNING "1000 prompts x 3 rounds"
  local ok=1
  for round in 1 2 3; do
    local f="$out/round-$(printf '%02d' "$round").jsonl"
    say "START $stage round=$round"
    docker exec "$CONTAINER" python3 -u -m sglang.bench_serving \
      --backend sglang-oai --base-url "$BASE" --dataset-name sharegpt \
      --dataset-path "$SG_DATA" --model /tmp/q38-target-model \
      --served-model-name "$MODEL" --sharegpt-context-len 131072 \
      --num-prompts 1000 --max-concurrency "$c" --request-rate inf \
      --warmup-requests $((2*c)) --seed 20260912 --disable-tqdm \
      --output-details --output-file "$f" \
      > "$out/round-$(printf '%02d' "$round").stdout.log" 2> "$out/round-$(printf '%02d' "$round").stderr.log"
    echo $? > "$out/round-$(printf '%02d' "$round").exitcode"
    [ "$(cat "$out/round-$(printf '%02d' "$round").exitcode")" = "0" ] || ok=0
    say "END $stage round=$round rc=$(cat "$out/round-$(printf '%02d' "$round").exitcode")"
  done
  mark "$stage" "$([ "$ok" = 1 ] && echo DONE || echo PARTIAL)" "3 rounds"
}

# ---------------- driver ----------------
say "=== SUITE START (pid $$) ==="
if ! wait_health; then mark suite BLOCKED "server not healthy"; say "FATAL server not healthy"; exit 1; fi
say "server healthy"
snapshot_server "$SUITE"

# --- priority order: 4K core -> gsm8k -> 64K -> 4K tail -> sharegpt -> other accuracy
run_perf_set 4096 1 8 16 32
run_acc_gsm8k
run_perf_set 65536 1 8 16 32
run_perf_set 4096 64 128
run_acc_evalscope acc_rwqa real_world_qa 100
for c in 1 4 16 32 64; do run_sharegpt "$c"; done
run_acc_evalscope acc_lcb hello 5

python3 - "$SUITE" <<'PY'
import json,pathlib,sys
suite=pathlib.Path(sys.argv[1]); rows=[]
for p in sorted(suite.glob("fixed-*/summary.json")):
    x=json.load(open(p))
    rows.append({"name":p.parent.name,"status":x.get("status"),
                 "out_tps_mean":(x.get("output_throughput_tps") or {}).get("mean"),
                 "out_tps_min":(x.get("output_throughput_tps") or {}).get("min"),
                 "out_tps_max":(x.get("output_throughput_tps") or {}).get("max"),
                 "ttft_p50_mean":(x.get("ttft_p50_s") or {}).get("mean"),
                 "ttft_p95_mean":(x.get("ttft_p95_s") or {}).get("mean"),
                 "tpot_p95_mean":(x.get("tpot_p95_ms") or {}).get("mean"),
                 "successful_rounds":x.get("successful_rounds")})
json.dump({"perf":rows,"stages":json.load(open(suite/"stage_status.json"))},
          open(suite/"suite_summary.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(rows,indent=2))
PY
say "=== SUITE END ==="
