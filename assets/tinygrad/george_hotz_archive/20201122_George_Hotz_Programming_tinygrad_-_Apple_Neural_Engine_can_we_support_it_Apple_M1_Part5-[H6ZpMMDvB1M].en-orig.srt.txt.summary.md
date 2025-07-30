# George Hotz's Apple Neural Engine (ANE) Exploration Summary

This transcript captures George Hotz reverse engineering the Apple Neural Engine (ANE) in M1 Macs to add support to his "tinygrad" project. Key findings include:

## Discoveries
- Found hidden "espresso" library that sits beneath Metal
- Discovered the private framework `apple_neural_engine` and related libraries:
  - `ANEServices` (interfaces with kernel)
  - `ANECompiler` (compiles neural network models)
- Identified `MLCDevice.ANE` in the MLCompute framework (not publicly exposed)
- Uncovered that the ANE uses "ZinIR" as its intermediate representation

## Technical Challenges
- Had to disable Apple Mobile File Integrity (amfi) via boot args to bypass security restrictions
- Struggled with code signing and entitlements needed to access ANE
- Needed to use low-level C APIs rather than Swift/Objective-C high-level interfaces

## Implementation Progress
- Successfully created an ANE device controller
- Got the kernel to accept ANE access requests
- Located operation codes and layer definitions in the compiler

## Next Steps
- Write a ZinIR compiler for tinygrad layers
- Use direct low-level access rather than high-level graph-based APIs
- Implement individual operations rather than whole-model compilation

George believes tinygrad could become the first library to directly leverage the ANE's "11 trillion operations per second" capability through direct programming rather than using Apple's abstractions.