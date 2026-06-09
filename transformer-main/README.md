# transformer

Caltech CS179 Transformer Project

# Background

I recommend you read [Attention Is All You Need](https://arxiv.org/abs/1706.03762) many times, this paper is critical to all transformer models.

We will be implementing the [Qwen2](https://arxiv.org/pdf/2407.10671) model, an open-weights LLMs.
Qwen2's architecture is copied from [Meta's Llama](https://arxiv.org/abs/2407.21783), except Qwen2 uses bias in the QKV matrices.
Both of these architectures implement [grouped-query attention](https://arxiv.org/abs/2305.13245),
[layer normalization](https://arxiv.org/abs/1607.06450v1) with zero mean,
[SwiGLU feed forward networks](https://arxiv.org/abs/2002.05202), and
[rotary positional embeddings](https://arxiv.org/abs/2104.09864).

# Implementation

We will only implement autoregressive decoding, which supports inference (generation) with one token at a time.
With single-token decoding, we avoid the masked attention operator which is the focus of most transformer optimization such as [FlashAttention](https://arxiv.org/pdf/2205.14135).
Additionally, we will not support batching.

We will use the [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) data type to store
model weights and key/value cache to reduce memory bandwidth relative to `float32`. However,
internally, we will use `float32` for accumulation within kernels, to minimize unnecessary floating-point rounding errors.

For simplicity, all tensors in this implementation will use row-major ordering, i.e. the memory is contiguous along the last dimension.
However, this layout is not optimal for performance is all cases.

# Model files

By default, the Qwen3.5 loader reads model files from the repo-local `models/Qwen3.5-4B` directory.
You can still override this with `TRANSFORMER_MODEL_DIR` or, for the Python reference, `--model-dir`.
The top-level `models/` directory is git-ignored because it contains local checkpoint files.

# Real world LLM inference

This project is missing many features in real-world production LLM inference, such as:
- Batching
- Prefill phase with masked matrix multiply
- Tensor cores
- Quantization
- Advanced model architectures, such as Mixture-of-Experts (MoE)
- Multi-GPU and multi-node parallelization
- KV cache offloading
- Sampling methods, such as Top-K

# Assignment
You will implement all kernels necessary for the LLM. Libraries are not allowed (such as cuBLAS, CUTLASS, cub, and thrust),
you must code all kernels from scratch.

## Part 1 (first week)

For the questions, cite sources you used. To submit, zip your repository to `~/lab5_2025_submission.zip`.

### Question 1.1 (5 points)
In this assignment, we will not be using tensor cores, because they require advanced data transfer layouts.
Instead, we will implement matrix-vector multiply with standard fused-multiply-add operators.
What is ratio of BF16 tensor core FLOPS to BF16 non-tensor core FLOPS on an A100-PCIE-40GB GPU?
Note: NVIDIA and AMD marketing both try to inflate their performance by measuring "sparse" tensor core operations, but nobody uses those.

The [specs](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) say that the BF16 tensor core dense performance is 312 TFLOPS and FP32 tensor core dense performance is 19.5 TFLOPS, for a ratio of 16.

### Question 1.2 (5 points)
What is the expected speedup of tensor cores vs non-tensor cores for matrix-vector multiplication on an A100-PCIE-40GB GPU?
Make an argument based on arithmetic intensity (FLOPS is not the whole story).
Assume the matrix and vector are read from off-chip memory.

The memory bandwidth is around 1555GB/s, which is (1555 * 10^9 bytes / 2 bytes) / 10^12 = 0.7775 BF16 floats. Since this is much less than 0.7775 < 312 and 19.5, the limiting factor is the loading of the data. Thus, there is no speedup at all for tensor vs non-tensor cores.

### Coding (80 points)
Implement GPU operators:
- ArgMax
- LayerNorm
- MatrixVectorMultiply
- RoPE
- SiLUMult

### Profiling (10 points)
Profile all your kernels with `ncu`, with input sizes matching what you'd expect for Qwen2 0.5B.
For each kernel, provide a screenshot and explain something interesting you noticed.
For example:
- Explain why your kernel is memory-bandwidth limited, latency/occupancy-limited, compute-limited, or limited by some other overhead.
- Explain why your kernel has suboptimal memory accesses, and a potential strategy to improve the kernel with expected performance increase.
- Explain which kernels are the most important to optimize, and which ones are less important.
- Explain how the performance would be different in another scenario (e.g. longer sequence length, larger model, increased batch size)
- Explain similarities across the kernels

- Argmax
  - compute bound (see speed of light)
  - probably because the reduction is the main part
  - also maybe bank conflict? shared memory request/response throughput are not even, not sure why that is
- Layernorm
  - split into sum of squares kernel (RMS) + layernorm kernel btw
  - numbers look a little funky, but sum of squares kernel should be compute-limited since it's a reduction just like argmax
  - layernorm should be memory-bound, decent L2 cache hit rate though so I think my gmem accesses are coalesced enough
- Matmul
  - limited by memory access (see speed of light + memory chart)
  - we load entire row of values per every output value from gmem
  - iirc there's an algorithm to "tile" the memory loading for matmuls and load tiles into shared memory and then compute then aggregate final values. should try that next
  - gmem reads are very coalesced tho (high L1 cache hit rate)
  - probably the most important since this definitely used the most compute
- Rope
  - memory-bound since computation is just angles + cosine
  - gmem acceses are pretty coalesced (high cache hit rate, see memory chart)
- Silu
  - memory-bound since computation is one multiplication + sigmoid
  - once again, gmem accesses are pretty coalesced (high cache hit rate, see memory chart)

## Part 2 (second week)

To submit, zip your repository to `~/lab6_2025_submission.zip`.

### Question 2.1 (3 points)
List all the matrix-vector multiplies in a Qwen2 0.5B layer, including the (M, K) dimensions of the matrix.
(Do not include grouped-query attention).

First, we have the QKV projections, which are 

- W_q: (m = queries_size = num_query_heads * query_dim = 14 * 64 = 896, k = hidden_dim = 896)
- W_k: (m = keys_size = num_kv_heads * key_dim = 2 * 64 = 128, k = hidden_dim = 896)
- W_v: (m = values_size = num_kv_heads * value_dim = 2 * 64 = 128, k = hidden_dim = 896)

Then, after attention, we have an output matrix that projects the attention output back into the same space as the hidden state, so the shape is W_o: (m = hidden_size, k = queries_size).

After the self attention layer, there's an MLP with 3 linear layers: 

- gate and up layers: (m = 4864, k = 896) 
- down layer: (m = 896, k = 4864).

### Question 2.2 (2 points)
Treating each query head as a row of a matrix, what are the dimensions of the matrix-matrix multiply in a
Qwen2 0.5B layer grouped-query attention operation? Assume current sequence length is 1234 tokens.

For each query head and kv head pair, we do a (num_query_heads, query_dim) @ (query_dim, 1234) matmul for the attention between the current token's queries and the past tokens keys for the given heads.

### Question 2.3 (5 points)
Assuming off-chip memory bandwidth is the limiting factor, what is the theoretical minimum inference latency (in ms)
for Qwen2 0.5B on an A100-PCIE-40GB, with BF16 weights? Assume small sequence length (i.e. KV cache size is negligible).

Since at the very least we have to load each of the weights for their corresponding ops e.g. matmuls, we need to load 0.5B * 2 bytes per bf16 = 1B bytes, and the A100-PCIE-40GB memory bandwidth is 1555GB/s, we have that the inference latency is 1 GB/1555 GB/s = 0.64ms.

### Question 2.4 (5 points)
Determine the sequence length at which the KV cache becomes non-negligible in terms of performance;
specifically, at what sequence length in Qwen2 0.5B would the KV cache become 10% the size of the model parameters?

Assuming that the KV cache and the model params are stored in the same type (e.g. all bf16 or float32), we find that for every token in each layer we need 2 key heads * 64 key dim = 128 floats and 2 value heads * 64 value dim = 128 floats => 256 floats total per token in each layer. Multiplying this by the number of layers (24), we get 24 * 256 = 6144. Then, 0.5B / (6144 * t) = 0.1 => ~8138 tokens.

### Coding (75 points)
Complete:
- GroupQueryAttention
  - Must use online numerically stable softmax, see section 3.1 of [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- Qwen2Layer
- Qwen2Model

You should not allocate or free any memory inside `Qwen2Model::forward`;
scratch space should be allocated only at model initialization, in constructors.

Test by running `./transformer`, and 100 tokens will be produced, matching the python reference implementation 

Once working, you can run `./transformer --interactive --max-seq-len 10000` to send messages with a chatbot interface.

### Profiling (10 points)

Once working, profile your implementation with:
```bash
ncu --set full --nvtx --nvtx-include last_token/ -c100 -o profile ./transformer
```
(may adjust -c parameter to number of kernels per layer)

How many microseconds per layer does your implementation take?
What is the slowest part of the layer and why?
Include screenshots of the something interesting you notice, and explain.

My implementation takes 2160 microseconds per layer. The slowest part is the SDPA kernel. Something interesting I noticed in the mem chart for the SDPA kernel (`profiling/sdpa.png`) was that I was getting a L2 cache hit rate of over 100%. Apparently this is because there isn't a specific hit/miss counter, and the hit rate is actuall derived from some other counters, like sector lookups for L2/DRAM.

## Assignment notes

- Your kernels must fully occupy the GPU when possible (i.e. do not launch with only 1 block, launch with many).
- Kernels should have optimal memory access (coalesced gmem, and no smem bank conflicts) when possible.
- Always use CUDA streams when launching kernels, such as:
  - `my_kernel<<<grid_dim, block_dim, 0, stream>>>(my_arg);`
- Use the test cases and python reference to check the correctness of your implementation.

## Debugging tips

- Add print statements in the python implementation and equivalents in the CUDA implementation, such as:
```python
print('after q proj:', queries[0, 0])
```
corresponding to
```c++
std::cerr << "after q proj: " << static_cast<float>(*static_cast<__nv_bfloat16*>(queries->data)) << std::endl;
```
and check when they diverge.

- All GPU memory is allocated with `cudaMallocManaged`, which allows you to access the GPU memory from the CPU.
  Therefore, with plain GDB, we can run:
  - `CUDA_LAUNCH_BLOCKING=1 gdb ./transformer`
  - Set breakpoints
  - Save a tensor to disk: `dump binary memory /tmp/queries.bin queries->data ((uint8_t*)queries->data)+queries->size`
  - Load the tensor in python: `torch.from_file('/tmp/queries.bin',size=head_size*num_query_heads,dtype=torch.bfloat16).reshape(num_query_heads, head_size)`

## Test cases
Run all test cases with:
- `cd build`
- `cmake --build .`
- `ctest`

For tests that are failing, you can run them individually to see which elements were incorrect, for example:
- `cd build`
- `cmake --build .`
- `./silumulttest`

Failing output:
```
difference at index 0: GPU calculated 10.875, CPU calculated 108.5
difference at index 1: GPU calculated -5.65625, CPU calculated 0.332031
difference at index 2: GPU calculated 7.3125, CPU calculated -76.5
...
```

Also, note that passing all test cases does not mean you will get an A.
The test cases only check for correctness, not for performance.
If your kernels have needless suboptimal memory access, poor occupancy, or other performance issues,
the tests will still pass, but you will not get a good grade.

## Author
Sam Foxman 2025
