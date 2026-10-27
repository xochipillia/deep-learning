#!/usr/bin/env bash
# Apply the official guided answer-repair (scripts/repair_parse_failures.py) to
# a COPY of the 2,048-token self-consistency records. The original records in
# results/self-consistency-2k/ are never modified.
#
# The official script repairs questions where *every* chain failed answer
# extraction: it appends "\n\nAnswer: " to each chain and forces one letter
# with guided decoding (A-E), then majority-votes the forced letters.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SRC=$(ls "$REPL"/results/self-consistency-2k/raw/*/candidates.jsonl)
OUT="$REPL/results/guided-repair"
rm -rf "$OUT"; mkdir -p "$OUT/raw"
cp "$SRC" "$OUT/raw/candidates.jsonl"
cp "$SRC" "$OUT/raw/candidates.input.jsonl"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$WORK"
python "$REPO/scripts/repair_parse_failures.py" --candidates "$OUT/raw/candidates.jsonl" \
  --dataset examsv_validation --model "$MODEL_ID" --dry-run > "$OUT/dry-run.log" 2>&1
cat "$OUT/dry-run.log"
if grep -q "No parse failures to repair" "$OUT/dry-run.log"; then
  python - "$OUT/run_meta.json" <<'EOF'
import json, sys
json.dump({"skipped": True, "reason": "official dry-run found no question with all chains failing extraction",
           "wall_clock_s": 0.0, "exit_code": 0}, open(sys.argv[1], "w"), indent=2)
EOF
  exit 0
fi
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 500 \
  > "$OUT/gpu-memory.csv" &
SMI=$!
START=$(date +%s.%N)
set +e
python "$REPO/scripts/repair_parse_failures.py" --candidates "$OUT/raw/candidates.jsonl" \
  --dataset examsv_validation --model "$MODEL_ID" > "$OUT/repair.log" 2>&1
RC=$?
set -e
END=$(date +%s.%N)
kill $SMI 2>/dev/null || true
cp "$OUT/repair.log" "$REPL/logs/guided-repair.log"
python - "$OUT/run_meta.json" "$START" "$END" "$RC" <<'EOF'
import json, sys
out, start, end, rc = sys.argv[1:]
json.dump({"skipped": False, "config": "configs/scaling/q35_sc_n8_2k.yaml + scripts/repair_parse_failures.py",
           "command": "python scripts/repair_parse_failures.py --candidates results/guided-repair/raw/candidates.jsonl --dataset examsv_validation --model Qwen/Qwen3.5-4B",
           "start_epoch": float(start), "end_epoch": float(end),
           "wall_clock_s": round(float(end) - float(start), 2), "exit_code": int(rc)}, open(out, "w"), indent=2)
EOF
tail -5 "$OUT/repair.log"
exit $RC
