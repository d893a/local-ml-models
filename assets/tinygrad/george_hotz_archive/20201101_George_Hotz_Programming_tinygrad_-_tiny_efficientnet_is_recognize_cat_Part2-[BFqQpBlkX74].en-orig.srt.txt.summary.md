# George Hotz's tinygrad Stream Summary

In this programming stream, George Hotz works on his neural network library **tinygrad**, focusing on two main objectives:

1. **Fixing EfficientNet image classification**
2. **Adding GPU support to tinygrad**

## EfficientNet Debugging

George begins by discussing a PR from Marcel Bischoff that significantly improved tinygrad's performance by replacing triple loops with `np.einsum`. He then tackles a bug where his EfficientNet implementation incorrectly classifies an image of a cat as a "coho" (salmon).

After investigation, he finds and fixes several issues:
- Missing skip connections in the MBConv blocks
- Incorrect application of swish activation on the output layer
- Problems with image category labels/mapping

Once fixed, the model successfully identifies various test images including cats, cars, and shoes.

## GPU Support Implementation

In the second part, George begins implementing GPU support for tinygrad using OpenCL:
- Creates `cuda()` and `cpu()` methods to move tensors between CPU and GPU
- Builds a dispatcher system to use different implementations based on tensor location
- Implements basic operations (`add`, `sum`) on the GPU
- Starts work on supporting backpropagation with GPU tensors

Throughout the stream, George emphasizes tinygrad's minimalist philosophy - implementing a complete neural network library in under 1000 lines while maintaining readability and elegance.

He closes by encouraging viewers to submit pull requests to implement additional GPU operations for tinygrad.