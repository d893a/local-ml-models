# George Hotz Programming Stream: tinygrad Updates and MCTS Search Optimization

## Key Highlights

- **tinygrad Progress**: George Hotz showcased recent developments in tinygrad, a lightweight machine learning framework competing with PyTorch and JAX. They've improved initialization methods, added comprehensive documentation, and enhanced kernel accuracy.

- **MCTS Search Optimization**: Much of the stream focused on optimizing Monte Carlo Tree Search (MCTS) for finding optimal kernel implementations. Hotz improved search speed from ~30 nodes/second to ~47 nodes/second by identifying and eliminating redundant operations.

- **Custom GPU Drivers**: Hotz discussed their custom AMD GPU driver that bypasses problematic parts of AMD's implementation by using PM4 instead of AQL to communicate directly with the hardware. They've upgraded their driver quality from "mediocre" to "acceptable."

- **TinyBox Hardware**: They're shipping TinyBox computers with multiple GPUs (shown with 6x 4090s), working through pre-orders with about 25 units shipped so far.

- **Search Efficiency**: The MCTS search demonstrated finding kernel implementations achieving 85-86% of theoretical maximum GPU performance. Hotz showed visualizations of how the search explores different optimization combinations.

- **Infrastructure Philosophy**: Emphasized "The Bitter Lesson" that general methods leveraging computation (like search and optimization) are ultimately most effective as hardware improves, rather than hand-crafted solutions.

- **Future Direction**: Plans to eventually make custom chips and potentially rewrite OpenPilot in tinygrad. The team is taking a divided approach to solving different performance challenges.

The stream combined live coding, performance optimization, and discussions about GPU hardware/software challenges in machine learning frameworks.