#!/usr/bin/env bash
set -u
ROOT=/data/qwen38-27b-k100ai-int8-opt/contest_20260923
BENCH="$ROOT/official/benchmark.py"
BASE="${BASE:-http://127.0.0.1:18580}"
MODEL=Q38-CONTEST-BEST-LAOUT3072-HCU67-20260923
OUTROOT="$ROOT/results/strongest-20260923"
C1="$OUTROOT/fixed-4k-1k-c1"

if [[ -f "$C1/runner.pid" ]]; then
  p=$(cat "$C1/runner.pid" || true)
  while [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; do sleep 5; done
fi

run_one() {
  local label=$1 ilen=$2 conc=$3
  local out="$OUTROOT/$label"
  mkdir -p "$out"
  if [[ -f "$out/summary.json" ]] && python3 - "$out/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("status")=="PASS" else 1)
PY
  then
    echo "SKIP_PASS $label"
    return 0
  fi
  local req=$((2*conc)); (( req < 8 )) && req=8
  echo "START $label input=$ilen concurrency=$conc requests=$req $(date -Is)"
  (hcu-snapshot || true) > "$out/hcu_snapshot_before.txt" 2>&1
  curl -sS "$BASE/v1/models" > "$out/models_before.json" 2>/dev/null || true
  docker inspect q38-contest-best-laout3072-hcu67-20260923 > "$out/container_inspect.json" 2>/dev/null || true
  python3 -u "$BENCH"     --base-url "$BASE"     --model "$MODEL"     --output-dir "$out"     --input-len "$ilen" --output-len 1024 --concurrency "$conc"     --warmup-rounds 2 --measure-rounds 3 --requests-per-round "$req"     --seed 20260912 --timeout 3600     > "$out/stdout.log" 2> "$out/stderr.log"
  rc=$?
  (hcu-snapshot || true) > "$out/hcu_snapshot_after.txt" 2>&1
  echo "$rc" > "$out/runner.exitcode"
  echo "END $label rc=$rc $(date -Is)"
  return 0
}

for c in 8 16 32 64 128; do run_one "fixed-4k-1k-c${c}" 4096 "$c"; done
for c in 1 8 16 32; do run_one "fixed-64k-1k-c${c}" 65536 "$c"; done

python3 - "$OUTROOT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
rows=[]
for p in sorted(root.glob("fixed-*/summary.json")):
    try:x=json.load(open(p))
    except Exception:continue
    rows.append({
      "name":p.parent.name,
      "status":x.get("status"),
      "throughput_mean":(x.get("output_throughput_tps") or {}).get("mean"),
      "ttft_p95_mean":(x.get("ttft_p95_s") or {}).get("mean"),
      "tpot_p95_mean":(x.get("tpot_p95_ms") or {}).get("mean"),
      "successful_rounds":x.get("successful_rounds"),
    })
out=root/"fixed_matrix_summary.json"
json.dump({"rows":rows},open(out,"w"),indent=2)
print(json.dumps({"rows":rows},indent=2))
PY
