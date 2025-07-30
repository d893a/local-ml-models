# George Hotz Programming Stream - Cherry Computer Project Summary

## Cherry Computer Overview
- George Hotz is developing "Cherry Computer," an AI accelerator hardware project aimed at competing with NVIDIA
- The goal is to create a more affordable, open-source alternative for neural network training
- The project tagline: "Don't you wish you had a choice"
- Hotz created a simple website for the company during the stream

## Technical Approach
- Based on RISC-V architecture with vector extensions
- Focuses on a single powerful core with a matrix multiplication engine rather than many small cores
- Plans to use TF32 floating-point format (19-bit version)
- The design includes large on-chip SRAM to reduce memory transfers
- Three-stage development plan:
  1. Initial FPGA implementation with 6.4 gigaflops
  2. "Cherry 2" with 4 teraflops, targeting $999 price point
  3. "Cherry 3" with petaflop performance, aimed to compete with high-end NVIDIA systems

## Development Process
- Hotz is working on emulator/simulator code in the tinygrad repository
- Reviewing RISC-V vector instruction specifications for implementation ideas
- Planning to implement both forward and backward passes for neural networks
- Aiming to participate in MLPerf benchmarks to demonstrate performance

## Business Philosophy
- Strong focus on creating actual value rather than extracting it
- Emphasis on building in-person teams rather than remote work
- Plans to make the Verilog code open source
- Goal to sell cards for $999-2000, significantly undercutting NVIDIA's pricing

## Timeline and Next Steps
- Initial FPGA implementation planned for near future
- Targeting ASIC development within about a year
- Looking for contributors to help with both hardware and software aspects
- Planning to hire people to work in-office in San Diego or LA

The stream mixed technical discussion with philosophical tangents about value creation, work ethics, and societal trends, reflecting Hotz's characteristic streaming style.