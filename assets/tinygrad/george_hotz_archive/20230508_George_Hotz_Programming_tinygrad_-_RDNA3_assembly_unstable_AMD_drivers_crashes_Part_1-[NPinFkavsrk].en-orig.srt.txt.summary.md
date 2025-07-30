# George Hotz Programming Session: AMD GPU Driver Issues Summary

This transcript documents George Hotz's livestream focused on working with AMD RDNA3 GPUs for his tinygrad project. The session highlights significant challenges with AMD's drivers and hardware.

## Key Issues Encountered

- **Driver Instability**: The AMD Rock M drivers repeatedly crash during basic operations
- **Performance Gap**: Only achieving ~15 teraflops on the AMD 7900 XTX versus the theoretical 61 teraflops
- **Kernel Panics**: Multiple system crashes requiring hard reboots
- **Inconsistent Behavior**: Issues across different kernels and computer setups

## Technical Explorations

- Attempting to write a custom assembler for RDNA3 instruction set
- Building custom kernels to test latest driver versions
- Disassembling shader code to understand optimization issues
- Swapping GPUs between systems to isolate hardware vs software problems

## Comparative Analysis

George compared driver stability across major GPU manufacturers:
- **Apple**: Most stable drivers, never crashes
- **NVIDIA**: Occasional issues but generally reliable
- **AMD**: Frequent crashes even with simple workloads

He noted a correlation between company market capitalization and driver stability, with Apple at the top, NVIDIA in the middle, and AMD at the bottom.

## Conclusion

George sets a deadline for returning the AMD cards if the issues can't be resolved, while making a plea to AMD:

1. Improve driver stability
2. Provide better documentation
3. Focus on making their GPUs viable for machine learning

Despite wanting to support AMD as an alternative to NVIDIA, the persistent driver issues make it difficult to use their hardware for serious development work. George expresses willingness to help improve the situation but needs AMD to demonstrate they care about fixing these fundamental problems.