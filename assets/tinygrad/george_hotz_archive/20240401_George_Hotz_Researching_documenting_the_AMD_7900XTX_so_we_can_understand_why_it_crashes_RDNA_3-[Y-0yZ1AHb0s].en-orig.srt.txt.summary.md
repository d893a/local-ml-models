# George Hotz AMD 7900XTX Research Stream Summary

George Hotz livestreams his efforts to document and understand the AMD 7900XTX GPU architecture, focusing on why it frequently crashes.

## Key Technical Exploration
- Uses a "kfd driver" backend in tinygrad that allows direct communication with GPU hardware
- Documents various internal GPU components:
  - MEC/ACE (Micro Engine Compute/Asynchronous Compute Engine) using RS64 architecture
  - SDMA (System DMA) and RLC (Run List Controller) using F32 architecture
  - MEEs (Micro Engine Scheduler)
  - 6 Shader Engines with 96 Compute Units total
- Examines GPU firmware, register mappings, and command dispatch mechanisms
- Uses utilities like UMR (User Mode Register) to inspect GPU state

## Challenges Encountered
- Frequent GPU crashes when querying certain registers
- Unreliable reset mechanisms when the GPU hangs
- Limited documentation from AMD
- No clear way to reliably debug GPU firmware

## Frustrations with AMD
- Lack of proper documentation for developers
- Promises to open-source components without following through
- No CI (Continuous Integration) testing for their GPUs
- Focusing on PR responses rather than technical solutions
- Declining market position against NVIDIA

## Future Plans
- Building comprehensive documentation to understand the architecture
- Mentions potential for creating his own GPU hardware in the future
- Offers $1000 bounty to anyone who can create a reliable GPU reset method

Throughout the stream, George frequently goes on tangents about corporate culture, technology philosophy, and other topics while continuously documenting his findings on the GPU architecture.