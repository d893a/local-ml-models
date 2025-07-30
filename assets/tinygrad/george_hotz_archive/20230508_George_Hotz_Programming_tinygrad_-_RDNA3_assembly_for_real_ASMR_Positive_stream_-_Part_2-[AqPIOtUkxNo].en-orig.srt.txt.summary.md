# George Hotz RDNA3 Assembly Stream Summary

In this programming stream, George Hotz works with AMD RDNA3 GPU assembly language as part of his tinygrad project. The session focuses on understanding and interfacing with AMD's open-source GPU drivers.

## Main Topics Covered

- Working with RDNA3 assembly code for AMD GPUs
- Developing a "sniffer" to intercept I/O controls between userspace and kernel
- Debugging signal handler issues when integrating with Python
- Analyzing DMA (Direct Memory Access) operations and queue types
- Parsing SDMA (System DMA) packets to understand memory transfers

## Technical Achievements

- Successfully ran custom GPU assembly code
- Implemented packet parsing for GPU operations
- Traced how buffers are transferred between CPU and GPU memory
- Analyzed how AMD runtime implements memory operations

## Challenges Faced

- Dealing with segmentation faults in the sniffer code
- Python signal handler interference
- Understanding complex AMD driver architecture
- Debugging GPU memory addressing

George humorously comments on AMD's open-source drivers, noting they "work more often than not" while acknowledging their occasional crashes. He approaches the work as a learning exercise to understand GPU internals with the ultimate goal of creating a simpler, more stable driver implementation.