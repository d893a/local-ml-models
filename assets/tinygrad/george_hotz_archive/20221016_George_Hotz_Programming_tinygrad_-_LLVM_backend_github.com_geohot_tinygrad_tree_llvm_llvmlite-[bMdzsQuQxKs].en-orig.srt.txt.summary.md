# George Hotz Programming Stream: LLVM Backend for tinygrad

This is a transcript of a coding stream where George Hotz (geohot) works on implementing an LLVM backend for tinygrad, a machine learning framework.

## Technical Focus

- **Goal**: Building an LLVM backend for tinygrad to improve memory access patterns
- **Implementation**: Creating basic operations (unary ops, binary ops, reduce ops) required by tinygrad
- **Debugging**: Significant time spent fixing segmentation faults caused by memory access issues
- **Shape Trackers**: Working on handling non-contiguous tensors with strides and offsets

## Key Progress

- Implemented several operations like add, subtract, multiply, divide, reciprocal, and relu
- Fixed memory corruption issues that caused segmentation faults (found an off-by-one error in loop bounds)
- Added proper handling for some shape tracker operations
- Several tests began passing after fixes

## Side Discussions

- Comments about a lawsuit with someone named Axel Nix
- Thoughts on intellectual property, patents, and copyright laws
- Discussions about AI's future impact on various industries
- Random Q&A with chat about technology, gaming, and other topics

The stream ends with some operations working but the reduce operations still unimplemented. George mentions potentially continuing the work in another stream.