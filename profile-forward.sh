#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
commit="$(git -C "$repo_root" rev-parse HEAD)"

exec ncu --set full --nvtx --nvtx-include last_token/ -c 200 -o profile-forward \
  "$repo_root/transformer-main/build/qwen35_forward_bench" \
  --git-commit "$commit" \
  "$@"
