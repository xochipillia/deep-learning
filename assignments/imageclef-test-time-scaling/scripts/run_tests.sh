#!/usr/bin/env bash
# Run the official repository unit tests (CPU only, fully mocked).
#   scripts/run_tests.sh <label>      e.g. before / after
set -euo pipefail
source "$(dirname "$0")/env.sh"
LABEL=${1:-before}
mkdir -p "$REPL/logs"
cd "$REPO"
set +e
python -m pytest tests/ -v -p no:cacheprovider --junitxml="$REPL/logs/tests-$LABEL.junit.xml" \
  > "$REPL/logs/tests-$LABEL.log" 2>&1
RC=$?
set -e
find "$REPO" -name __pycache__ -type d -prune -exec rm -rf {} +
tail -5 "$REPL/logs/tests-$LABEL.log"
echo "repo clean: $(test -z "$(git status --porcelain)" && echo yes || git status --porcelain)"
exit $RC
