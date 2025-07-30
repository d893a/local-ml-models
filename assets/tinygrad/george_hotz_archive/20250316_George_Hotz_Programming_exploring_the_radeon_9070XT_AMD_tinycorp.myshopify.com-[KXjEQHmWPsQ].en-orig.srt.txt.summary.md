This is a transcript from a George Hotz livestream where he explores AMD's new Radeon RX 9070 XT GPU. Here's a summary of the key points:

## Main Content

**GPU Testing & Bug Discovery:**
- George tests the new AMD RX 9070 XT GPU with TinyGrad (his machine learning framework)
- Discovers a significant bug in AMD's HIP runtime where large kernel launches silently fail
- The bug manifests when GPU grid sizes exceed 65,536 in the first dimension - it should throw an error but instead produces incorrect results
- Compares behavior between the new GPU and older 7900 XTX, finding the new chip is slightly faster (78 vs 69 tokens/second with optimizations)

**Technical Details:**
- Updates TinyGrad to support the new GPU architecture (gfx1201)
- Tests various ML workloads including Llama inference
- Identifies that AMD's compiler/runtime lacks proper assertions for invalid launch parameters
- Implements a CPU validation feature in TinyGrad to catch GPU computation errors

## Side Commentary

**Economic/Political Rants:**
- Extended discussion criticizing fiat currency and advocating for the gold standard
- Claims current monetary system is a "scam" causing widespread impoverishment
- Discusses inflation graphs and argues everything should be cheaper due to technological progress

**Industry Commentary:**
- Criticizes OpenAI's direction under Sam Altman
- Praises AMD hardware while noting their software issues
- Mentions being an AMD investor ($250k investment)
- References competition with NVIDIA and supply chain issues

**Personal Notes:**
- Streams from Hong Kong
- Mentions receiving $500k worth of AMD hardware for testing
- Plans to release TinyBox v2 with new GPUs

The stream demonstrates both technical GPU debugging and George's characteristic tangential commentary on broader topics.

---

# George Hotz Radeon 9070XT Stream Summary

George Hotz explores the new AMD Radeon 9070XT GPU while working with the tinygrad framework. The main technical focus is troubleshooting compatibility issues between tinygrad and the new GPU.

## Key Technical Discoveries

- George identifies a bug affecting large kernel launches on the new GPU that doesn't affect older AMD GPUs (7900 XTX)
- The issue occurs because the 9070XT silently fails when exceeding grid size limits (65536) rather than throwing an error
- Through systematic debugging, he creates a minimal failing example that reproduces the issue
- After finding the bug, LLaMa model inference works correctly on the new GPU
- Performance testing shows the 9070XT getting 69 tokens per second initially, improving to 78 with optimizations

## Hardware Context

- George mentions having received two MI300X machines from AMD (high-end compute GPUs)
- He discusses AMD's hardware quality versus their software implementation
- Notes he's now an AMD investor ("we now shill for AMD")
- Shows rdna4 architecture information and discusses instruction sets

## Other Notes

- George implements a GPU vs CPU validation feature in tinygrad to help identify inconsistencies
- Discusses driver quality improvements in AMD's software stack
- The stream includes tangents about monetary theory, inflation, and the gold standard
- Stream ends with the bug successfully identified and resolved

The key conclusion is that the AMD GPU has solid performance but suffers from poor error handling in the driver, with silent failures instead of helpful error messages.