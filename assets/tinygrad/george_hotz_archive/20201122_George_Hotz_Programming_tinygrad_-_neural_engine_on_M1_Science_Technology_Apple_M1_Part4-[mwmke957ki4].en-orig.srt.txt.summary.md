# George Hotz Programming Stream Summary

This stream featured George Hotz working on TinyGrad with a goal of supporting Apple's Neural Engine on the M1 chip.

## Key Achievements

1. **Fixed IPython/Jupyter on Apple Silicon**
   - Debugged a segmentation fault using fault handler
   - Discovered the issue was related to AppNap on Apple Silicon
   - Submitted a fix to make IPython notebooks work properly

2. **Created PyTorch-free Model Loading**
   - Built a custom loader to use PyTorch models without requiring PyTorch
   - Successfully parsed the pickle format used by PyTorch
   - Got efficient-net running with the custom loader

3. **GPU Support on M1**
   - Fixed PyOpenCL for ARM64 architecture
   - Got TinyGrad running on the M1 GPU
   - Noted it wasn't as fast as the AMD GPU in his 16" MacBook

## Neural Engine Investigation

- Explored Apple's ML frameworks including ML Compute and Core ML
- Struggled with Swift's verbosity when trying to use Apple's native frameworks
- Attempted to understand how Apple interacts with the Neural Engine
- Concluded he would need to disable System Integrity Protection to use dtrace for further investigation

## M1 MacBook Air Impressions

- **Positives**: Impressed by the M1 chip performance, no fan, battery life
- **Negatives**: Criticized the speakers ("terrible"), webcam quality, and screen bezels
- Mentioned he had the 7-core GPU variant and joked about wishing he had the 8-core
- Overall seemed impressed with the chip itself despite criticisms of the hardware

The stream ended with plans to continue investigating the Neural Engine in a future session after disabling System Integrity Protection.