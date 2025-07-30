# George Hotz TinyGrad Optimization Session Summary

This transcript captures George Hotz live-coding session focused on hand-optimizing a ResNet-50 neural network implementation in TinyGrad, his minimalist deep learning framework.

## Technical Focus

George demonstrates TinyGrad's new abstractions for neural network optimization:

- Shows how to access and manipulate the execution schedule of ResNet-50
- Filters and examines the 59 operations in the model
- Uses the linearizer to convert operations into executable code
- Benchmarks individual kernels to identify performance bottlenecks
- Experiments with manual optimization techniques including:
  - Loop unrolling (upcasting)
  - Tensor reshaping
  - Dimension permutation
  - Local work group size adjustments

## Key Achievements

- Identifies and improves performance of specific kernels
- Demonstrates how to compare performance with/without tensor cores
- Achieves up to 139 gigaflops on one previously underperforming kernel
- Showcases TinyGrad's clean abstraction layers and debugging capabilities

## Philosophy

George emphasizes several key principles:

- Understanding every component rather than relying on black-box libraries
- Building generic optimizers over hand-tuned solutions
- Creating clean abstractions that work together seamlessly
- Keeping the entire framework compact (~4,416 lines of code)

The session demonstrates how TinyGrad provides visibility into the entire neural network execution pipeline from high-level PyTorch-like code down to generated GPU kernels, all within a small codebase.