# George Hotz Tinygrad Stream Summary

George Hotz livestreamed his work on tinygrad, a minimalist deep learning framework he's developing with a focus on keeping the codebase under 1000 lines while maintaining functionality.

## Key Accomplishments

- Successfully ran the OpenPilot autonomous driving model in tinygrad
- Added ONNX model import capability to tinygrad
- Got EfficientNet image classification working (correctly identifying images of a chicken and a car)
- Implemented several new operations in tinygrad including:
  - eLU activation
  - Multi-cat (concatenation)
  - Flatten
  - Clip
  - Transpose
  - Batch normalization

## Technical Insights

- Demonstrated tinygrad's architecture with three abstraction levels:
  1. High-level Tensor operations
  2. Mid-level MLOps (memory allocation and derivatives)
  3. Low-level LLOps for different accelerators (CPU, GPU, PyTorch)
- Converted matrix multiplications to 1x1 convolutions for simplification
- Found that nearly all ONNX models use strided max pooling, which wasn't supported yet
- Discussed future optimizations:
  - Making tinygrad lazy (only computing when explicitly needed)
  - Implementing kernel fusion to reduce GPU dispatch overhead
  - Using convolutions as a base operation for implementing other operations

## Results

The stream demonstrated tinygrad's capability to run real-world models with minimal code. George visualized the computational graphs of the models, showing how complex networks like OpenPilot and EfficientNet can be represented using tinygrad's simplified operations.

George described tinygrad as essentially a "RISC instruction set for neural networks" that can be optimized at a low level, potentially making it faster than other frameworks for inference and possibly training.