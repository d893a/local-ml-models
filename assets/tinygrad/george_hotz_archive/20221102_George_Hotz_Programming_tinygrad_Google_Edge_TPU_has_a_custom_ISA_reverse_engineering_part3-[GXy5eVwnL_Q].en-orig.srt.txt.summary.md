# George Hotz Edge TPU Reverse Engineering Session Summary

## Overview
In this live-coding session, George Hotz works on reverse engineering the Google Edge TPU's custom instruction set architecture (ISA). The Edge TPU is a machine learning accelerator chip with minimal public documentation about its internal workings.

## Key Discoveries

### Basic Architecture
- The Edge TPU uses 16-byte (256-bit) instructions
- Found evidence of a "bundle ALU move I instruction" in documentation
- Identified the Edge TPU has a scalar core with 32 registers

### Instruction Set Progress
George successfully reverse engineered several scalar operations:
- **Basic ALU operations**: add, subtract, AND, OR, XOR
- **Comparison operations**: equal, not equal, less than, greater than
- **Shift operations**: arithmetic and logical shifts
- **Move immediate** instructions for loading values

### Technical Achievements
1. Created a working program that reliably divides numbers by 2
2. Built a primitive disassembler for the Edge TPU instructions
3. Discovered how to read from the scalar register file
4. Figured out how to set breakpoints in the code
5. Identified predicate registers for conditional execution
6. Parsed instruction fields (prefixes, immediate values, register addresses)

### Methodology
George used a systematic approach:
- Compiled simple TensorFlow operations to the Edge TPU
- Compared generated code between different operations
- Binary searched through instructions to identify critical ones
- Used breakpoints to examine register values
- Modified instructions and observed behavior changes

## Challenges
- Limited documentation on the Edge TPU architecture
- Complex instruction encoding with multiple fields
- Understanding different processing units (scalar, vector, TTU)
- Deciphering memory transfers and I/O operations

## Conclusion
While significant progress was made on understanding the scalar operations, more work is needed to understand vector operations and control flow instructions. George ended the session with a clearer understanding of the Edge TPU's scalar unit, but still has more to discover about its overall architecture.