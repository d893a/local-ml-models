# George Hotz tinygrad Optimization Livestream Summary

This is a transcript of George Hotz (geohot) livestreaming his work on tinygrad, a lightweight machine learning framework he's developing. The main focus is optimizing tinygrad's performance on Apple M1 chips to beat PyTorch.

## Key Technical Issues and Discoveries

1. **FMA Fusion Problem**: Most of the stream is spent debugging why LLVM isn't properly fusing multiply-add operations. After extensive troubleshooting, he discovers the solution was simply changing optimization level from `-O3` to `-O2`.

2. **Performance Comparison**: George runs benchmarks comparing tinygrad vs PyTorch:
   - Some operations in tinygrad are already faster (like `add_square` and `mulsum`)
   - Matrix multiplications are ~20x slower in tinygrad
   - Convolutions are ~2x slower

3. **AMX Coprocessor**: George identifies that PyTorch's superior matrix multiplication performance comes from using Apple's AMX coprocessor instructions, which he plans to implement in tinygrad.

4. **Vectorization Issues**: Another problem is that max operations aren't being properly vectorized, hurting performance.

## Development Goals

George has a roadmap written on his whiteboard:
1. Beat PyTorch CPU on M1 at everything
2. Release fastest Stable Diffusion on M1
3. Beat PyTorch CUDA on 3080 Ti on everything

He mentions that implementing AMX support could potentially make Stable Diffusion run 3x faster on M1 than with PyTorch.

## Tools and Techniques

- Using LLVM for code generation
- Testing multiple backends (LLVM, OpenCL, NumPy)
- Cache optimization strategies
- Compiler flags and optimization levels
- Using Godbolt Compiler Explorer for debugging

The stream demonstrates the nitty-gritty details of performance optimization work and compiler interactions, showing both the frustrations and breakthroughs that come with low-level optimization.