# George Hotz tinygrad Stream Summary

## Technical Work
- George implemented a "bitcast" functionality in tinygrad to allow proper type conversion between different data types without changing bit representation
- He debugged issues with llama2 model loading, particularly with B-float16 support
- Refactored the code to make bitcast a flag in the existing cast operation rather than a separate operation, simplifying the implementation
- Tested the functionality with llama2 model loading and generation

## tinygrad Meeting & Project Updates
- Daily team meeting discussed various pull requests and bounties
- Mentioned a $400 bounty for improving matrix reduction optimization
- Discussed the goal of beating A100 GPU performance with the "Tiny Box" hardware
- Announced moving meetings from daily to weekly (Mondays only) due to inefficiency
- Multiple bounties available for training ML models and improving performance

## Tiny Box Hardware
- George is developing custom hardware called "Tiny Box" with AMD GPUs
- Mentioned it has 144GB RAM (vs 80GB in A100) and more FLOPS
- Working on a custom case design with comma's mechanical engineering team
- Plans to deploy Tiny Boxes in comma's compute cluster

## Other Commentary
- Expressed frustration about code complexity and the need for refactoring
- Shared thoughts on management challenges and remote work dynamics
- Discussed future vision of potentially building custom AI chips
- Mentioned an upcoming debate with Connor Leahy scheduled for the next day
- Recommended Orange Pi over Raspberry Pi for robotics applications

The stream demonstrated George's hands-on approach to both coding and project management, while showcasing the ongoing development of tinygrad as an alternative to PyTorch.