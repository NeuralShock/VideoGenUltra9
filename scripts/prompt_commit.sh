#!/usr/bin/env bash

set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: scripts/prompt_commit.sh \"<prompt text>\""
  exit 1
fi

PROMPT_TEXT="$*"
LOG_FILE="logs/chat_history.md"
TIMESTAMP="$(date +"%Y-%m-%d %H:%M:%S %Z")"
BRANCH="$(git branch --show-current)"

mkdir -p logs

if [ ! -f "${LOG_FILE}" ]; then
  cat > "${LOG_FILE}" <<'EOF'
# Chat History

Running log of prompt-driven changes and local commits.
EOF
fi

git add -A

if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

CHANGED_FILES="$(git diff --cached --name-only)"

{
  echo
  echo "## ${TIMESTAMP}"
  echo "- Prompt: ${PROMPT_TEXT}"
  echo "- Branch: ${BRANCH}"
  echo "- Files:"
  while IFS= read -r file; do
    [ -n "${file}" ] && echo "  - ${file}"
  done <<< "${CHANGED_FILES}"
} >> "${LOG_FILE}"

git add "${LOG_FILE}"
git commit -m "${PROMPT_TEXT}"

echo "Committed with prompt message and updated ${LOG_FILE}."
