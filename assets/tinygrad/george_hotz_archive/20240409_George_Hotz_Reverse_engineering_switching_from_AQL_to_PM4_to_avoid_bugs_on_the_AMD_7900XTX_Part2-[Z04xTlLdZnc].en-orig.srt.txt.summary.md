# George Hotz GPU Reverse Engineering Session Summary

George Hotz spent this session investigating bugs on the AMD 7900XTX GPU by examining whether switching from AQL (Asynchronous Queue Language) to PM4 (Packet Manager 4) command packets could avoid issues. Key discoveries include:

1. **Register Investigation**: Used UMR (User Mode Register) to examine GPU registers and understand how commands are processed.

2. **AQL vs PM4**: Explored how PM4 packets might bypass complexity in the AQL implementation, potentially avoiding bugs. PM4 appears more straightforward than the AQL abstraction layer.

3. **Major Breakthrough**: Discovered that AMD's "RS64" firmware is actually just RISC-V code. This makes understanding and potentially modifying the firmware much easier.

4. **Code Analysis**: Loaded the firmware into Ghidra and began mapping GPU registers and command structures.

5. **Command Structure**: Found that the compute dimensions and dispatch commands in PM4 are similar to NVIDIA's approach.

6. **Banking System**: Discovered how the GPU uses register banking for accessing different hardware blocks.

7. **Theory**: The complexity of the AQL queuing/scheduling mechanism is likely responsible for bugs, with PM4 potentially offering a more direct approach.

8. **Future Direction**: With the RISC-V discovery, there's potential to better understand the firmware or even create custom firmware that doesn't use the problematic AQL implementation.

The session involved extensive debugging, register dumping, and firmware analysis with tools like UMR and Ghidra, providing significant insights into how AMD's GPU architecture processes commands.