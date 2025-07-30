# George Hotz's Nvidia IOCTL Exploration Session - Technical Summary

George Hotz is working on switching tinygrad from AMD to Nvidia GPUs due to driver instability issues with AMD. The session focuses on creating a lower-level interface to Nvidia GPUs by sniffing IOCTL calls.

## Key Technical Points

### New NV Interface Development
- Creating a layer called "NV" below CUDA that directly talks to Nvidia hardware
- Users would run tinygrad with `NV=1` instead of `CUDA=1`
- Goal: Replace entire Nvidia userspace stack with open implementation

### IOCTL Sniffing Implementation
- Building `nvy_octl.py` to intercept Nvidia driver IOCTLs
- Extracting command definitions from Nvidia open-source kernel module headers
- Mapping IOCTL numbers to symbolic names and parsing their parameters
- Key IOCTLs examined: `NV_ESC_CARD_INFO`, `RM_ALLOC`, memory mapping operations

```python
# Sample of IOCTL interception code
if nr in nvs:
    print(nvs[nr])
```

### Observed Nvidia Driver Behavior
- Nvidia uses few IOCTLs during computation (mostly just for setup)
- Demonstrates intercepting PyTorch's CUDA initialization IOCTLs
- Explores PTX and SASS code generation via debug flags

### tinygrad Architecture
- Currently at HSA level, moving to HSA KMT (kernel mode driver) level
- Demonstrates operation graphs and kernel scheduling visualization
- Shows how different backends (CUDA, OpenCL, PTX) can be used interchangeably

### Technical Vision
- Create a "sovereign stack" independent of vendor libraries
- Focus on "better scheduling" as key to outperforming Nvidia
- Three operation types: reduce ops, element-wise ops, and movement ops
- Moving toward static scheduling for all ML compute operations

### Hardware Considerations
- Discusses potential to use large BAR (Base Address Register) to map entire GPU memory
- Possibility of DMA transfers directly between GPUs
- Importance of small data types (fp4/fp8) for compute efficiency

### Product Impact
- "Tiny Box" product switching from AMD to Nvidia (4x4090 GPUs)
- Priced at $649 for 16 PFLOPS (FP16)
- Still plans to get AMD working for MLPerf benchmarks

The session demonstrates a deep technical exploration of GPU driver interfaces with the goal of building an independent, high-performance ML infrastructure stack.