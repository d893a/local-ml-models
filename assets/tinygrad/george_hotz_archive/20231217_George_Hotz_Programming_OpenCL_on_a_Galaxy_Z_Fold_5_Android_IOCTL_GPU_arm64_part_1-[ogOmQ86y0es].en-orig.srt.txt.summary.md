# George Hotz OpenCL Hacking Stream Summary

In this stream, George Hotz works on accessing and modifying GPU operations on a Samsung Galaxy Z Fold 5 Android phone at a lower level than the standard OpenCL API.

## Technical Accomplishments

- Connected to the phone via SSH using Termux (a Linux environment for Android)
- Created a Python-based hook for the `ioctl` system call that intercepts communications between OpenCL and the GPU driver
- Implemented a function "trampoline" in ARM64 assembly to redirect calls to his custom code
- Successfully intercepted and displayed GPU allocation commands in real-time
- Worked through memory protection, instruction cache, and function pointer manipulation challenges

## Goals and Motivation

- Build a simpler, more direct GPU driver that bypasses OpenCL's abstractions
- Understand exactly what happens when allocating GPU memory (noting that 16-byte allocations were being rounded up to 4KB)
- Create a foundation for a more universal GPU driver approach that could work across different hardware

## Philosophical Tangents

The stream included extensive commentary on:
- The importance of understanding systems at deeper levels
- Criticism of abstraction-heavy programming and developers who don't understand underlying technology
- Discussions about technological progress, politics, and economic trends
- Thoughts on motivation, skill, and hiring philosophy

The work demonstrates Hotz's approach of bypassing high-level abstractions to gain direct control over hardware, consistent with his work on tinygrad (his lightweight neural network framework).