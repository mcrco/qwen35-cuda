#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
commit="$(git -C "$repo_root" rev-parse HEAD)"

exec "$repo_root/build/qwen35_module_bench" \
  --git-commit "$commit" \
  "$@"
