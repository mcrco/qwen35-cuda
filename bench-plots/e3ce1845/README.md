Optimized after first implementations. Improvements made were:

- parallelized prefix sum/search for sampling
- parallelized linear attention gated delta net using strat 3 in cuh file
- flash attention for gqa
- block reduction for mat vec mult
