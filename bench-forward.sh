#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
commit="$(git -C "$repo_root" rev-parse HEAD)"
default_prompt="Explain CUDA kernels briefly."
args=("$@")
has_prompt=false
for arg in "${args[@]}"; do
  if [[ "$arg" == "--prompt" || "$arg" == --prompt=* ]]; then
    has_prompt=true
    break
  fi
done
if [[ "$has_prompt" == false ]]; then
  args=(--prompt "$default_prompt" "${args[@]}")
fi

"$repo_root/build/qwen35_forward_bench" --git-commit "$commit" --prefill "${args[@]}"
uv run python "$repo_root/pyref/hf_bench.py" --git-commit "$commit" --prefill "${args[@]}"
