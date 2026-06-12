# CUDA Qwen3.5 Inference

CS179 final project for extending the course transformer implementation toward
Qwen3.5 autoregressive inference.

## Build CUDA Code

```bash
cd transformer-main
cmake -S . -B build
cmake --build build -j
```

## Install Python Reference Environment

```bash
cd transformer-main/pyref
uv venv --python 3.13
uv pip install torch==2.7.0 safetensors==0.5.3 transformers==4.51.3
```

Run the reference implementation with:

```bash
uv run python main.py --model-dir ../../models/Qwen3.5-0.8B --prompt a --max-new-tokens 100 --dtype float32
```

## Install Plotting Environment

```bash
cd plots
uv venv --python 3.13
uv pip install -r requirements.txt
```

Regenerate benchmark plots with:

```bash
uv run python plot_benchmarks.py --results-dir ../bench-results --out-dir ../bench-plots --replot-all
```

## Install Model Files

Model directories belong under `models/`. This directory is git-ignored because
checkpoints are large.

Example small checkpoint used for local testing:

```bash
mkdir -p models/Qwen3.5-0.8B
curl -L -o models/Qwen3.5-0.8B/config.json https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json
curl -L -o models/Qwen3.5-0.8B/tokenizer.json https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json
curl -L -o models/Qwen3.5-0.8B/model.safetensors.index.json https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors.index.json
curl -L -o models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors
```

The C++ executable defaults to `models/Qwen3.5-4B`. To run a different local
checkpoint:

```bash
export TRANSFORMER_MODEL_DIR="$PWD/models/Qwen3.5-0.8B"
```

## Project Description and Features

This project extends the Qwen 2.5 0.5B implementation so that it can load and
run Qwen3.5-style models.

The most important new computation is the Gated DeltaNet layer. Instead of
storing all previous keys and values and doing full softmax attention at every
layer, the linear-attention layers keep a recurrent state that is updated as
tokens stream through the model:

```text
prediction_t = state_{t-1} * key_t
delta_t = value_t - prediction_t
state_t = decay_t * state_{t-1} + step_t * outer(delta_t, key_t)
output_t = state_t * query_t
```

The implementation includes:

- loading different Qwen3.5 model sizes, currently 0.8B, 4B, and 9B shapes, for both bf16 and float32 (although I didn't do explicit instantions for bf16 since my RTX 2070 Super doesn't support it),
- a Python reference implementation for Qwen3.5 pieces,
- CPU kernels for debugging,
- CUDA matrix-vector multiplication,
- zero-centered RMSNorm,
- gated RMSNorm and row L2 normalization,
- RoPE changes for the Qwen3.5 shapes,
- flash-attention-style grouped-query attention,
- CUDA Gated DeltaNet / linear-attention recurrent update,
- temperature sampling,
- an end-to-end autoregressive decoding path,
- an interactive chat interface from the course starter code.

## Expected Results and Screenshots

Run all test cases:

```bash
cd transformer-main/build
ctest --output-on-failure
```

Run the CUDA executable:

```bash
cd transformer-main/build
TRANSFORMER_MODEL_DIR=../../models/Qwen3.5-0.8B ./transformer --max-seq-len 100
```

Run the interactive chat interface:

```bash
cd transformer-main/build
TRANSFORMER_MODEL_DIR=../../models/Qwen3.5-0.8B ./transformer --interactive --max-seq-len 10000
```

Run the Python reference on the same checkpoint:

```bash
cd transformer-main/pyref
uv run python main.py --model-dir ../../models/Qwen3.5-0.8B --prompt a --max-new-tokens 100 --dtype float32
```

The deterministic CUDA path and Python reference should be compared on the same
checkpoint, dtype, prompt/token stream, and temperature.

Screenshot placeholder: add screenshot(s) of generated text here.

## Performance Analysis

Note: These were run on my PC, specs are AMD Ryzen 7 3700X (16 core) @ 3.600GHz with 32GB DDR4 and NVIDIA GeForce RTX 2070 SUPER with 8GB.

Benchmarks are run through `bench-module.sh`, which records the current git
commit in each result file:

```bash
cd transformer-main
cmake --build build -j
../bench-module.sh --module all --preset qwen35-4b --json ../bench-results
```

The benchmark JSON files are in `bench-results/`. The plot directories in
`bench-plots/` are named by git commit. The latest checked-in benchmark is:

```text
bench-results/module-all-e3ce18459048.json
bench-plots/e3ce1845/
```

The latest benchmark README records the main improvements after the initial
implementation:

- parallelized prefix sum/search for sampling,
- parallelized linear attention Gated DeltaNet using strategy 3 in the header,
- flash attention for GQA,
- block reduction for matrix-vector multiplication.

Nsight Compute screenshot placeholder: add forward-profile screenshots showing
memory coalescing, bank conflicts, and other kernel details here.

## Potential Improvements

- Fuse small kernels such as normalization, gating, and activation where possible
  to reduce launch overhead.
- Improve matvec tiling.
- Add a real prefill implementation for multi-token prompts instead of only
  single-token decoding.
- Add batching, top-k/top-p sampling, and more production-style cache management.
- Add true quantized weight loading and inference. I was not able to get to this
  within the project time constraints.
