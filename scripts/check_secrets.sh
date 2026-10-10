#!/usr/bin/env bash
# Fail if any tracked (default) or staged (--staged) file contains something that looks like an API key.
set -euo pipefail
pattern='sk-(proj-)?[A-Za-z0-9_-]{20,}'
if [[ "${1:-}" == "--staged" ]]; then
  files=$(git diff --cached --name-only --diff-filter=ACM)
else
  files=$(git ls-files)
fi
[[ -z "$files" ]] && exit 0
if printf '%s\n' "$files" | tr '\n' '\0' | xargs -0 grep -nIE "$pattern" -- 2>/dev/null; then
  echo "ERROR: possible API key found (see above). Remove it before committing." >&2
  exit 1
fi
exit 0
