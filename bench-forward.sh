#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
commit="$(git -C "$repo_root" rev-parse HEAD)"

"$repo_root/build/qwen35_forward_bench" --git-commit "$commit" "$@"
uv run python "$repo_root/pyref/bench.py" --git-commit "$commit" "$@"
