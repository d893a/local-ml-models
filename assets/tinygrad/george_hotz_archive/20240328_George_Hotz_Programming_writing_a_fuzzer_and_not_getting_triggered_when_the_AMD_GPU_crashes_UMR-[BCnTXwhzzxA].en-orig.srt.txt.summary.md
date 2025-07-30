# George Hotz Programming Stream Summary

This transcript captures George Hotz working on debugging AMD GPU crashes while developing a product called "Tiny Box." Key points:

## Main Technical Issue
- Working on debugging persistent crashes with AMD GPUs
- Creating a "fuzzer" to identify and reproduce GPU crashes
- Frustrated by the difficulty in diagnosing GPU issues without proper documentation

## Major Discovery
- Found an existing open-source tool called "UMR" (User Mode Register debugger) that allows access to AMD GPU registers
- UMR provides visibility into GPU register states before and after crashes
- This discovery was transformative for his debugging process

## Business Context
- Planning to ship Nvidia GPUs in commercial "Tiny Box" products (more reliable)
- Considering also shipping AMD-based "Red Tiny Box" as a developer platform after finding UMR

## Remaining Concerns
- Still wants AMD to open source their firmware
- Needs better documentation for the registers
- Expresses frustration that AMD representatives never mentioned UMR in communications

## Meta Observations
- Discusses the difficulty of finding information online even when it's public
- Critiques modern search engines for their inability to surface technical documentation
- Alternates between technical work and philosophical discussions about technology

The stream shows Hotz's transition from frustration to cautious optimism about working with AMD GPUs after discovering the UMR tool that had been available all along.