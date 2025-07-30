# George Hotz Programming Mixtral on a Tinybox: Stream Summary

## OnePlus Rant
- George is upset with OnePlus after bricking his OnePlus Open phone by setting fastboot to partition B
- OnePlus refused to honor their 15-day return policy for the $1500 phone
- Unlike previous OnePlus models, newer models (after OnePlus 9) don't have publicly available QDL (recovery) files
- Key takeaway: "Never buy anything from the OnePlus store, always buy on Amazon" for better consumer protection

## Technical Work: Implementing Mixtral on Tinybox
- Successfully implemented Mixtral (mixture-of-experts model) across multiple GPUs
- Distributed the model by putting the attention mechanism on GPU 0 and experts across GPUs 1-4
- Implemented the expert routing mechanism that selects which 2 experts (out of 8) to use per token
- Debugged issues with AMD GPU memory copying from disk
- Refactored the implementation to be more elegant using partial functions

## Multi-GPU Implementation Details
- Created a "mixture feed forward" class to handle expert selection
- Used device management to distribute computation across GPUs
- Demonstrated cross-GPU tensor operations with minimal overhead
- Fixed issues with read-only memory views during loading

## Performance and Observations
- Generated text successfully despite debugging challenges
- Achieved about 90ms per token inference time (10 tokens/second)
- Identified overhead in Python implementation that could be optimized
- Demonstrated tinygrad's capability to handle complex multi-GPU workloads

## Other Topics
- Announced several $1000 bounties for tinygrad improvements
- Discussed future work on GPU sharding for linear layers and batches
- Emphasized the importance of good customer support in business
- Shared philosophy on implementing ML systems and open-source development

The stream demonstrated both technical ML system implementation and George's approach to software development, while repeatedly warning viewers about his negative OnePlus customer experience.