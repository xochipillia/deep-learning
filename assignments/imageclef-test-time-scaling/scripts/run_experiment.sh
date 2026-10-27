#!/usr/bin/env bash
# Run one official config through the official runner (scripts/experiment.py)
# on the seed-42 stratified 200-question subset, unchanged.
#
#   scripts/run_experiment.sh <result-name> <config path relative to repo> [extra experiment.py args]
#
# Example:
#   scripts/run_experiment.sh self-consistency-2k configs/scaling/q35_sc_n8_2k.yaml
#   scripts/run_experiment.sh smoke-test/zero-shot configs/baselines/q35_zero_shot.yaml --end-idx 10
#
# Outputs (in $REPL/results/<result-name>/):
#   raw/<run_id>/candidates.jsonl   official per-question records (source of truth)
#   raw/<run_id>/config.yaml        resolved config written by the official logger
#   raw/subset_<wall>.json          subset manifest written by the official runner
#   gpu-memory.csv                  nvidia-smi samples (500 ms) during the run
#   run_meta.json                   wall-clock times, exit code, command
#   experiment.log                  full runner log (also copied to $REPL/logs/)
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=$1; CONFIG=$2; shift 2
OUT="$REPL/results/$NAME"
SCRATCH="$WORK/scratch/$NAME"
CFG_COPY="$REPL/configs/${CONFIG#configs/}"   # configs/<group>/<file>.yaml
rm -rf "$SCRATCH" "$OUT/raw"; mkdir -p "$SCRATCH" "$OUT/raw" "$REPL/logs" "$(dirname "$CFG_COPY")"
cp "$REPO/$CONFIG" "$CFG_COPY"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # resolve to the pinned cached revisions

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 500 \
  > "$OUT/gpu-memory.csv" &
SMI=$!
CMD=(python "$REPO/scripts/experiment.py" --config "$REPO/$CONFIG" --subset-override "$SUBSET_SIZE" "$@")
START=$(date +%s.%N)
set +e
( cd "$SCRATCH" && "${CMD[@]}" ) > "$OUT/experiment.log" 2>&1
RC=$?
set -e
END=$(date +%s.%N)
kill $SMI 2>/dev/null || true
cp -r "$SCRATCH/runs/." "$OUT/raw/" 2>/dev/null || true
cp "$OUT/experiment.log" "$REPL/logs/$(echo "$NAME" | tr / _).log"
python - "$OUT/run_meta.json" "$START" "$END" "$RC" "$CONFIG" "${CMD[*]}" <<'EOF'
import json, sys
out, start, end, rc, cfg, cmd = sys.argv[1:]
json.dump({"config": cfg, "command": cmd.replace("/home/sagemaker-user/work/", "$WORK/"),
           "start_epoch": float(start), "end_epoch": float(end),
           "wall_clock_s": round(float(end) - float(start), 2), "exit_code": int(rc)},
          open(out, "w"), indent=2)
EOF
echo "[$NAME] exit=$RC wall=$(python -c "print(round($END-$START,1))")s"
grep -E "=== .*%|Failed on|Traceback" "$OUT/experiment.log" | tail -5 || true
exit $RC
