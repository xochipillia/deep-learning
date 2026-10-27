#!/usr/bin/env bash
# Pre-handoff hygiene scan of the replication/ package. Exits non-zero on any finding.
# Checks: credentials/tokens, AWS account ids, personal paths/e-mails, dataset
# files, model weights/caches, large files, presentation files.
set -uo pipefail
PKG=${1:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$PKG"
FAIL=0
hit() { echo "FINDING: $1"; FAIL=1; }
# 1. Credentials / tokens
grep -rInE '(AKIA|ASIA)[A-Z0-9]{16}' . && hit "AWS access key id"
grep -rInE 'aws_secret_access_key|aws_session_token|SecretAccessKey"|SessionToken"' --exclude=scan_package.sh . && hit "AWS secret/session token"
grep -rInE '\bhf_[A-Za-z0-9]{30,}\b' . && hit "HuggingFace token"
grep -rInE 'WANDB_API_KEY=[a-f0-9]{20,}|ghp_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY' . && hit "API key / private key"
# 2. Account ids and personal identifiers (pass extra patterns via SCAN_EXTRA_PATTERNS, '|' separated)
grep -rInE '\b[0-9]{12}\b' --include='*.md' --include='*.txt' --include='*.sh' --include='*.py' --include='*.json' --include='*.yaml' . \
  | grep -vE 'results/|examples/|885854791233|137914896644|542918446943' && hit "possible 12-digit AWS account id"
grep -rInE '/Users/|C:\\\\Users' --exclude=scan_package.sh . && hit "personal home path"
grep -rInoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}' --exclude-dir=results --exclude-dir=examples . | grep -v 'git@github' && hit "e-mail address"
[ -n "${SCAN_EXTRA_PATTERNS:-}" ] && grep -rInE "$SCAN_EXTRA_PATTERNS" --exclude=scan_package.sh . && hit "personal identifier"
# 3. Dataset files, weights, caches, presentations, large files
find . -type f \( -name '*.parquet' -o -name '*.arrow' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
  -o -name '*.pth' -o -name '*.gguf' -o -name '*.jpg' -o -name '*.png' -o -name '*.pptx' -o -name '*.key' -o -name '*.pdf' \) | grep . && hit "dataset/weight/image/presentation file"
find . -type d \( -name 'hf-cache' -o -name 'huggingface' -o -name 'models--*' -o -name 'datasets--*' -o -name '.venv' -o -name 'venv' \) | grep . && hit "cache or venv directory"
find . -type f -size +50M | grep . && hit "file larger than 50 MB"
[ $FAIL = 0 ] && echo "scan: no findings"
exit $FAIL
