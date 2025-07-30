# George Hotz on tinygrad, tiny corp, and Democratizing AI Computing

In this Latent Space podcast episode, George Hotz (geohot) discusses his work on tinygrad, tiny corp, and his vision for making AI compute more accessible.

## tinygrad: A Minimalist ML Framework

Hotz describes tinygrad as a "RISC" approach to machine learning frameworks - focusing on simplicity with just 25 operations compared to the 250+ in frameworks like PyTorch or XLA. Originally limited to 1,000 lines of code (now ~2,800), tinygrad emphasizes readability and minimal complexity while providing several advantages:

- **Lazy evaluation** for operation fusion, combining operations like `a * b + c` into single kernels instead of separate memory operations
- **Built-in debugging** with `graph=1` to visualize operations and `debug=2` to show detailed performance metrics
- **Better cross-platform support**, particularly for AMD GPUs and Qualcomm devices

Currently, tinygrad is about 5x slower than PyTorch on NVIDIA hardware but competitive or faster on platforms like Qualcomm GPUs, where it powers models in Hotz's comma.ai driving system.

## tiny corp: Commoditizing Petaflops

Hotz founded tiny corp to ensure AI compute remains accessible, fearing potential restrictions on ML hardware. The company plans to:

1. Build "Tiny Box" - a high-performance AI computer for home use that can fit under a desk and run silently
2. Focus on optimizing for flops per dollar and flops per watt
3. Eventually create custom chips designed specifically for ML workloads

He explains that current AI hardware often has unnecessary complexity from CPU-style features like branch predictors and warp schedulers, which are irrelevant for the static execution patterns of neural networks.

## Hardware Challenges and Strategy

Hotz details the challenges of building multi-GPU systems:
- Physical constraints of fitting 4-slot GPUs in standard chassis
- Power limitations (six 350W GPUs exceed standard outlets)
- PCIe bandwidth constraints vs. NVLink (60GB/s vs. 600GB/s)
- Cooling and noise considerations

He's engaged with both AMD and NVIDIA, highlighting how AMD initially delivered poor driver experiences but improved after direct communication with CEO Lisa Su.

## Future of AI and Computing

Hotz shares several provocative perspectives:
- Training smaller models for longer may prove more effective than massive models
- Current chatbots suffer from "mode collapse" due to training with categorical cross-entropy on internet data
- Future AI systems should have human-computer merging capabilities without invasive interfaces
- The AI alignment problem is misframed - the real issue isn't AI aligning with companies, but companies aligning with people

## tiny corp's Approach to Building

Hotz has implemented an unconventional hiring process based on bounty completion rather than traditional interviews. He believes traditional programming challenges and technical screens are ineffective predictors of contribution quality.

He views AI tools as augmenting human capabilities rather than replacing them, particularly for tasks "above the API line" like coding, while expressing confidence that in approximately 20 years, machines will ultimately replace most human tasks.