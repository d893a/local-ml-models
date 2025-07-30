# George Hotz's AMD GPU Driver Exploration Stream Summary

In this technical stream, George Hotz (geohot) attempts to work directly with AMD's low-level GPU interfaces by bypassing the standard userspace drivers. His goal is to create a more efficient interface for his machine learning framework tinygrad.

## Key Technical Work

- Explores the KFD (Kernel Fusion Driver) interface to directly control AMD GPUs
- Successfully allocates GPU memory using direct IO controls
- Attempts to create command queues but consistently encounters GPU crashes
- Builds cleaner abstractions around the raw IO control interfaces

## Major Issues Encountered

- Simple operations (creating queues) repeatedly crash the GPU requiring system reboots
- Testing on updated driver versions shows the same problems
- Issues appear to be systemic rather than isolated bugs

## Key Criticisms of AMD

- Firmware and drivers have fundamental reliability issues
- Communication with AMD described as corporate speak without substance
- Claims AMD adds mitigations rather than fixing root causes
- Speculates there might be hardware-level issues that AMD won't acknowledge
- Frustrated by perceived lack of interest in achieving "greatness"

## Technical Insights

- Discusses graph compilation and optimization for ML frameworks
- Explains how tinygrad represents operations as graphs that get lowered to hardware command queues
- Considers future directions for hardware acceleration
- Contrasts AMD's approach with NVIDIA's more reliable implementation

## Conclusion

Hotz ultimately abandons the effort due to persistent crashes, concluding AMD GPUs should be avoided for serious machine learning work due to reliability issues. He plans to focus on NVIDIA GPUs for the "tiny box" systems he's developing.