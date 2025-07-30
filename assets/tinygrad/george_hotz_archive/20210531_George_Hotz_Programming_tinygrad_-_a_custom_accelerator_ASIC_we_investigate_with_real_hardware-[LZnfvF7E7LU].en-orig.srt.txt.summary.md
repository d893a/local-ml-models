# George Hotz FPGA Programming Stream Summary

George Hotz (geohot) livestreamed his exploration of creating a custom accelerator for tinygrad, his neural network framework. The stream follows his journey working with FPGA hardware and low-level programming.

## Key Accomplishments

* Successfully connected and programmed an Artix 7 100T FPGA board
* Set up an open-source toolchain for FPGA development on macOS
* Implemented a RISC-V processor on the FPGA
* Created custom code to control the board's LEDs
* Implemented UART communication
* Compiled C code to run on the custom RISC-V processor

## Hardware Discussion

George analyzed the limitations of his FPGA board, particularly its limited memory:
* 256MB of DDR3 RAM with limited bandwidth
* Around 600KB of SRAM (block RAM)
* Not enough memory to fit even a single batch of EfficientNet B0

He compared his affordable board to more expensive options, noting the trade-offs between cost and capabilities.

## Future Directions

Toward the end of the stream, George discussed:
* Potentially creating a custom ASIC for tinygrad through services like Skywater/chipIgnite
* The possibility of offering a USB accelerator for tinygrad users
* Jokingly proposed creating an "80-core CPU" to attract venture capital investment

## Side Notes

Throughout the stream, George shared his thoughts on Silicon Valley culture, programming tools, and occasional tangents about delivery services and his music on SoundCloud. The stream demonstrated his hands-on approach to understanding hardware acceleration for machine learning from first principles.