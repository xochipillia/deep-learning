#!/usr/bin/env bash
# Environment creation: clone the official repo at the pinned commit, create a
# Python 3.12 venv with the repo's requirements, pre-fetch the pinned model and
# dataset revisions into the HF cache, and record system information.
set -euo pipefail
SYS_PY=${SYSPY:-$(command -v python3)}   # interpreter that hosts uv (outside the venv)
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK" "$REPL/environment" "$REPL/logs"

# 1. Official repository at the preferred commit (fallback: current main).
if [ ! -d "$REPO/.git" ]; then git clone "$REPO_URL" "$REPO"; fi
cd "$REPO"
git fetch --quiet origin
if git cat-file -e "${REPO_COMMIT}^{commit}" 2>/dev/null; then
  git checkout --quiet "$REPO_COMMIT"
else
  echo "Preferred commit $REPO_COMMIT not found; using origin/main" >&2
  git checkout --quiet origin/main
fi
git rev-parse HEAD > "$REPL/environment/repo-commit.txt"
test -z "$(git status --porcelain)" || { echo "repo not clean" >&2; exit 1; }

# 2. Virtual environment (uv for fast, reproducible resolution).
"$SYS_PY" -m pip install --quiet --user uv
UV="$SYS_PY -m uv"
[ -x "$VENV/bin/python" ] || $UV venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
$UV pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt"
$UV pip freeze --color never --python "$VENV/bin/python" > "$REPL/environment/requirements-lock.txt"

# 3. Pin model + dataset revisions in the local HF cache. Experiments later run
#    with HF_HUB_OFFLINE=1 so the official loaders resolve to exactly these.
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo, rev, kind in [(os.environ["MODEL_ID"], os.environ["MODEL_REVISION"], "model"),
                        (os.environ["DATASET_ID"], os.environ["DATASET_REVISION"], "dataset")]:
    p = snapshot_download(repo_id=repo, revision=rev, repo_type=kind)
    print(kind, repo, rev, "->", p)
EOF
# The official loaders request the default revision ("main"). In offline mode
# the hub cache resolves "main" through refs/main, which snapshot_download(sha)
# does not write; point it at the pinned SHA (equal to main on 2026-10-25).
for spec in "models--${MODEL_ID//\//--} $MODEL_REVISION" "datasets--${DATASET_ID//\//--} $DATASET_REVISION"; do
  set -- $spec; mkdir -p "$HF_HOME/hub/$1/refs"; printf '%s' "$2" > "$HF_HOME/hub/$1/refs/main"
done
# Warm the datasets cache (Arrow conversion) while online, at the pinned revision.
python - <<'EOF'
import os
from datasets import load_dataset
ds = load_dataset(os.environ["DATASET_ID"], revision=os.environ["DATASET_REVISION"])
print({k: v.num_rows for k, v in ds.items()})
EOF

# 4. System information.
bash "$REPL/scripts/system_info.sh" > "$REPL/environment/system-info.txt"
cat "$REPL/environment/system-info.txt"
