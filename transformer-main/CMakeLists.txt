cmake_minimum_required(VERSION 3.27)
project(transformer CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit)

get_filename_component(TRANSFORMER_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

add_compile_options(-Wno-changes-meaning)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

add_library(transformer_lib
        src/vendor/argparse.hpp
        src/vendor/safetensors.hh
        src/vendor/safetensors_impl.cpp
        src/CudaBuffer.cuh
        src/CudaBuffer.cu
        src/ErrorCheck.h
        src/qwen35/Qwen35Config.h
        src/qwen35/Qwen35Types.cuh
        src/cpu_ops/BufferOps.cuh
        src/cpu_ops/BufferOps.cu
        src/cpu_ops/ArgMax.cuh
        src/cpu_ops/ArgMax.cu
        src/cpu_ops/LayerNorm.cuh
        src/cpu_ops/LayerNorm.cu
        src/cpu_ops/Math.cuh
        src/cpu_ops/MatrixVectorMultiply.cuh
        src/cpu_ops/MatrixVectorMultiply.cu
        src/cpu_ops/SiLUMult.cuh
        src/cpu_ops/SiLUMult.cu
        src/cpu_ops/RoPE.cuh
        src/cpu_ops/RoPE.cu
        src/cpu_ops/GroupQueryAttention.cuh
        src/cpu_ops/GroupQueryAttention.cu
        src/cpu_ops/LinearAttention.cuh
        src/cpu_ops/LinearAttention.cu
        src/cpu_ops/Sampling.cuh
        src/cpu_ops/Sampling.cu
        src/qwen35/Qwen35Layer.cuh
        src/qwen35/Qwen35Layer.cu
        src/qwen35/Qwen35Model.cuh
        src/qwen35/Qwen35Model.cu
        src/qwen35/Qwen35Loader.h
        src/qwen35/Qwen35Loader.cu
        src/HostBuffer.cpp
        src/HostBuffer.h
        src/BPE.cpp
        src/BPE.h
        src/gpu_ops/ArgMax.cu
        src/gpu_ops/ArgMax.cuh
        src/gpu_ops/GpuFloat.cuh
        src/gpu_ops/LayerNorm.cu
        src/gpu_ops/LayerNorm.cuh
        src/gpu_ops/RoPE.cu
        src/gpu_ops/RoPE.cuh
        src/gpu_ops/MatrixVectorMultiply.cu
        src/gpu_ops/MatrixVectorMultiply.cuh
        src/gpu_ops/SiLUMult.cu
        src/gpu_ops/SiLUMult.cuh
        src/gpu_ops/Sampling.cu
        src/gpu_ops/Sampling.cuh
        src/gpu_ops/GroupQueryAttention.cuh
        src/gpu_ops/GroupQueryAttention.cu
        src/gpu_ops/LinearAttention.cuh
        src/gpu_ops/LinearAttention.cu
        )

target_link_libraries(transformer_lib PRIVATE CUDA::cudart CUDA::nvtx3)
target_compile_definitions(transformer_lib PRIVATE TRANSFORMER_REPO_ROOT="${TRANSFORMER_REPO_ROOT}")

add_executable(transformer
        src/vendor/argparse.hpp
        src/main.cu
)
target_link_libraries(transformer PRIVATE transformer_lib)

add_executable(qwen35_forward_bench
        src/vendor/argparse.hpp
        src/bench/qwen35_forward_bench.cu
)
target_link_libraries(qwen35_forward_bench PRIVATE transformer_lib CUDA::nvtx3)
target_compile_definitions(qwen35_forward_bench PRIVATE
        TRANSFORMER_REPO_ROOT="${TRANSFORMER_REPO_ROOT}")

add_executable(qwen35_module_bench
        src/vendor/argparse.hpp
        src/bench/qwen35_module_bench.cu
)
target_link_libraries(qwen35_module_bench PRIVATE transformer_lib CUDA::nvtx3)
target_compile_definitions(qwen35_module_bench PRIVATE
        TRANSFORMER_REPO_ROOT="${TRANSFORMER_REPO_ROOT}")

# tests
enable_testing()

add_executable(bpetest ./src/tests/BPETest.cu)
target_link_libraries(bpetest PRIVATE transformer_lib)
add_test(NAME bpetest COMMAND bpetest)

add_executable(argmaxtest ./src/tests/ArgMaxTest.cu)
target_link_libraries(argmaxtest PRIVATE transformer_lib)
add_test(NAME argmaxtest COMMAND argmaxtest)

add_executable(samplingtest ./src/tests/SamplingTest.cu)
target_link_libraries(samplingtest PRIVATE transformer_lib)
add_test(NAME samplingtest COMMAND samplingtest)

add_executable(matvecmulttest ./src/tests/MatVecMultTest.cu)
target_link_libraries(matvecmulttest PRIVATE transformer_lib)
add_test(NAME matvecmulttest COMMAND matvecmulttest)

add_executable(layernormtest ./src/tests/LayerNormTest.cu)
target_link_libraries(layernormtest PRIVATE transformer_lib)
add_test(NAME layernormtest COMMAND layernormtest)

add_executable(ropetest ./src/tests/RoPETest.cu)
target_link_libraries(ropetest PRIVATE transformer_lib)
add_test(NAME ropetest COMMAND ropetest)

add_executable(silumulttest ./src/tests/SiLUMultTest.cu)
target_link_libraries(silumulttest PRIVATE transformer_lib)
add_test(NAME silumulttest COMMAND silumulttest)

add_executable(gqatest ./src/tests/GQATest.cu)
target_link_libraries(gqatest PRIVATE transformer_lib)
add_test(NAME gqatest COMMAND gqatest)
