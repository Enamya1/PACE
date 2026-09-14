#!/usr/bin/env bash
#
# PACE pre-commit secret scanner.
# Fails the commit if AWS-key-like patterns appear anywhere in staged text files.
#
# Usage:   bash scripts/check_secrets.sh <file1> <file2> ...
# Can also be run manually:  bash scripts/check_secrets.sh  (scans entire repo)

set -euo pipefail

PATTERNS=(
    'AKIA[0-9A-Z]{16}'
    'ASIA[0-9A-Z]{16}'
    'aws_secret_access_key'
    'AWS_SECRET_ACCESS_KEY'
    'aws_access_key_id'
    'AWS_ACCESS_KEY_ID'
    '(?i)aws(.{0,20})?(?-i)['\''\"][0-9a-zA-Z/+]{40}['\''\"]'
)

if [ "$#" -eq 0 ]; then
    TARGETS="."
    echo "[check_secrets] Scanning entire working tree..."
else
    TARGETS="$*"
    echo "[check_secrets] Scanning staged files: $# file(s)"
fi

FOUND=0
for PATTERN in "${PATTERNS[@]}"; do
    # grep exits 1 when no match, which is fine; we only care about exit 0 (match found)
    if grep -r -n -E "$PATTERN" $TARGETS --include='*.py' --include='*.yaml' --include='*.yml' \
        --include='*.env*' --include='*.cfg' --include='*.ini' --include='*.toml' \
        --include='*.md' --include='*.txt' --include='*.sh' --include='*.json' \
        --exclude-dir='.git' --exclude-dir='venv' --exclude-dir='.venv' \
        --exclude-dir='__pycache__' --exclude-dir='.mypy_cache' --exclude-dir='.pytest_cache' \
        2>/dev/null; then
        echo ""
        echo "ERROR: AWS-key-like pattern detected: $PATTERN"
        echo "       Remove the secret or use an encrypted env file / AWS vault instead."
        echo "       Per project privacy rules, NO secrets may be committed."
        FOUND=1
    fi
done

if [ "$FOUND" -ne 0 ]; then
    exit 1
fi

echo "[check_secrets] OK - no suspicious patterns found."
