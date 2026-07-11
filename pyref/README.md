# Python reference implementation

PyTorch reference implementation of the transformer model.

## Setup

From the repository root:

```bash
uv sync
uv run python pyref/main.py
```

Dependencies are declared in the root `pyproject.toml`. The PyTorch wheel uses
CUDA 11.8 for compatibility with Pascal (`sm_61`) GPUs.
