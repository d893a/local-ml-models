# George Hotz Programming Stream: tinygrad LLVM Speed Optimization Summary

George Hotz streamed from Hong Kong, focusing on improving tinygrad's CPU performance. Key points:

## Speed Optimization Work
- Created documentation about different aspects of performance in tinygrad
- Found and fixed alignment issues preventing use of wider YMM vector registers
- Improved performance from ~29 gigaflops to ~50 gigaflops (PyTorch gets ~75)
- Identified memory alignment and coalescing as critical performance factors

## Technical Concepts Covered
- Memory alignment importance for SIMD operations
- LLVM code generation and optimization challenges
- Beam search for finding optimal kernel implementations
- L1 cache optimization using "locals"
- Register pressure and stack spilling issues

## Project Context
- Developing a custom AMD GPU driver to replace the unstable official driver
- Working on "Tiny Box" systems with multiple GPUs
- Acquiring new NVIDIA 5090 GPUs at high prices

## Community Engagement
- Created bounties for performance improvements in tinygrad
- Encouraged meaningful contributions to the project
- Emphasized that contributors don't need to be experts, just able to add value

George ended the stream encouraging viewers to contribute to tinygrad in meaningful ways, stating he'll do more streams if the community shows genuine engagement with the project.