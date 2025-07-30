# George Hotz Multi-GPU Programming with HIP Stream Summary

George Hotz is working on optimizing multi-GPU performance for his deep learning framework "tinygrad" using AMD's HIP (AMD's CUDA equivalent). The stream shows his real-time debugging and optimization process.

## Key Technical Findings:

1. **Initial Problem**: GPU-to-GPU memory transfers were extremely slow (~3.9 GB/s)

2. **P2P Access Discovery**: Enabling peer-to-peer access between GPUs allowed direct memory access between them:
   ```python
   # Enable P2P access between devices
   hip.check(hip.hipDeviceEnablePeerAccess(d0, 0))
   ```

3. **Synchronization Challenge**: Creating an effective synchronization mechanism between GPUs proved difficult due to cache coherency issues

4. **GPU Cache Problem**: The critical issue was that the L2 cache on AMD GPUs doesn't properly flush when doing cross-device memory operations

5. **Solution Found**: Setting `HSA_DISABLE_CACHE=1` environment variable fixes the issue but disables the entire L2 cache

## About Tiny Corp and Tinygrad:

- George explains his "tinygrad" framework (22,000 GitHub stars) which uses minimal operations:
  - Element-wise ops
  - Reduce ops
  - Movement ops

- His company "Tiny Corp" sells "Tiny Box" - a system with 6-7 AMD 7900 XTX GPUs with full PCIe connectivity between all GPUs

- Tiny Box is positioned as a more affordable alternative to NVIDIA H100 systems

The stream ends with George highlighting AMD's disappointing official solution for the GPU cache coherency problem - completely disabling the L2 cache.