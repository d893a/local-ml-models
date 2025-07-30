# George Hotz's Mixture of Experts Implementation Stream Summary

George Hotz worked on implementing a mixture of experts (MoE) model in tinygrad, similar to DeepSeek's approach, targeting AMD hardware. MoE models are efficient because they only activate a subset of parameters for each input token.

## Technical Progress:

1. Started implementing the mixture of experts architecture by loading and transforming weights from a model called ALOM-1B-7B
2. Spent time debugging tensor operations and model architecture components:
   - Created more efficient tensor representation for experts
   - Implemented the routing mechanism that selects which experts to activate
   - Fixed issues with normalization and expert selection

3. Encountered and debugged a subtle bug related to rotary positional embeddings (RoPE):
   - Discovered that Hugging Face stores weights in a permuted format compared to other implementations
   - The bug was causing incorrect token predictions in the model
   - Fixed by properly handling the dimension permutation

4. Identified next steps:
   - Clean up the implementation to be more concise
   - Fix a memory leak
   - Make the implementation faster
   - Get the model working in the JIT (just-in-time compilation)
   - Create a more efficient tokenizer implementation

## Hardware Notes:
- AMD is sending George Mi300X hardware to test this implementation
- He plans to optimize the model for fast inference on AMD GPUs

The stream demonstrated both the technical challenges of implementing modern AI architectures and the importance of understanding subtle differences between frameworks when porting models.