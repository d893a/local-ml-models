# George Hotz Programming Stream Summary: Training ImageNet with tinygrad

George Hotz (geohot) livestreamed his work on optimizing tinygrad, his lightweight machine learning framework. Key points from the stream:

## Main Focus
- Attempting to train ImageNet with tinygrad
- Implementing and optimizing a ResNet-like architecture for CIFAR-10
- Comparing performance against a highly-optimized PyTorch implementation

## Technical Work
- Implemented a "SpeederResNet" neural network architecture
- Fixed multiple bugs in the tinygrad optimizer
- Added various optimizations that improved performance by ~6.5x
- Used "chaopt" (kernel optimization system) with different levels to accelerate operations
- Added progress bars for downloads using tqdm

## Performance Analysis
- Initially tinygrad was ~64x slower than PyTorch
- After optimizations, reduced the gap to ~9-10x slower
- Achieved around 2-3 teraflops with optimizations
- Identified that PyTorch's implementation was achieving ~24 teraflops

## Code Quality Discussions
- Emphasized writing clean, concise code over verbose implementations
- Reviewed and critiqued pull requests for the project
- Discussed proper code refactoring versus "code golf"
- Examined the generated GPU kernels and their performance characteristics

## Project Status
- tinygrad has grown from 1000 lines to ~1780 lines of code
- Being used in the comma.ai OpenPilot system for inference
- George mentioned the "Tiny Corp" is seeking contributors, offering "unpaid internships"
- Looking for sponsorships and considering selling merchandise to fund development

The stream showed significant progress on making tinygrad more competitive with established frameworks, though more work remains to match PyTorch's performance.