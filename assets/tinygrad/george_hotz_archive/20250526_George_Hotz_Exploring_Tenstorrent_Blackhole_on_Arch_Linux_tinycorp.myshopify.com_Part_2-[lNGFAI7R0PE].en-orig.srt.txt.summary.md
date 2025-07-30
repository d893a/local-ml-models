# George Hotz's Tenstorrent Blackhole Evaluation - Summary

## Hardware Exploration
George Hotz explores the Tenstorrent Blackhole, an AI accelerator with a novel architecture featuring:
- A mesh of "tensics" cores with explicitly managed memory
- 32x32 matrix operations as base compute units
- Different RISC processors controlling various functions
- Alternative approach to the traditional GPU architecture

## Critical Software Assessment
Hotz is highly critical of Tenstorrent's software stack:

- **Excessive Abstraction**: "Please bro, one more stack. This stack will be good, I promise"
- **Fragmented Ecosystem**: Multiple competing frameworks (TTorch, TT Forge, TTNN, TT Metalium)
- **Poor Developer Experience**: Complex C++ dependencies, Docker containers, incomplete documentation
- **Fundamental Architecture Issues**: Operations hard-coded rather than dynamically generated
- **Limited Fusion**: No ability to combine operations that should be executed together

## Comparison with Other Hardware
- **AMD**: "AMD has come a long way" - Now works well with PyTorch after years of improvement
- **NVIDIA**: "NVIDIA drivers just work" - Acknowledges quality despite closed-source nature
- **Tiny Corp's Approach**: Demonstrates tinygrad running efficiently on AMD with minimal setup

## Technical Philosophy
Hotz advocates for simplicity and elegance in systems design:

> "You can't build a castle on a shit swamp. As long as you still have things that look like SFPU_LU as an LLK, you are guaranteed to lose."

His recommended architecture:
1. Simple C API runtime (like CUDA)
2. Graph compiler for memory placement and fusion
3. Clean front-end abstractions

## Industry Criticism
Throughout the stream, Hotz critiques broader tech industry issues:
- "What's come out of Silicon Valley in the last 15 years? Nothing"
- Describes companies passing "fake money" between each other without creating real value
- Criticizes "goal-oriented behavior" without deeper purpose

## Advice to Tenstorrent
Hotz's blunt recommendations:

> "If you want to get acquired/become scam IP licensing co, I can't help you. If you want to win AI compute, read on."

1. Simplify to three abstraction layers
2. Build a real dataflow graph compiler
3. Leverage their hardware's unique programmability advantage
4. Stop chasing NVIDIA on NVIDIA's terms

The stream reveals Hotz's technical brilliance and strong opinions on hardware/software co-design, showing his vision for AI computing that emphasizes simplicity, comprehensibility, and performance.

---

# George Hotz Stream Summary: Tenstorrent Blackhole Exploration Part 2

This lengthy stream covers George Hotz's continued exploration of the Tenstorrent Blackhole AI accelerator card, mixed with philosophical discussions and comparisons to other hardware platforms.

## Key Technical Points

**Tenstorrent Hardware Analysis:**
- George successfully gets some basic kernels running on the Tenstorrent card after moving it to another room due to noise
- He critiques their software stack complexity, specifically calling out the multi-layered abstraction (TTN, TT-Metalium, LLK, etc.)
- Main criticism: Too many abstraction layers between PyTorch and hardware, unlike AMD's approach of directly cloning CUDA

**AMD vs Tenstorrent Comparison:**
- Demonstrates AMD's dramatic improvement - a simple ResNet model runs immediately with just `pip install rocm`
- Praises AMD for copying Nvidia's approach rather than reinventing everything
- Shows AMD achieving ~85% of theoretical memory bandwidth on LLM inference

**Software Philosophy:**
- Advocates for simpler, more direct hardware abstractions
- Criticizes the trend of adding more abstraction layers instead of solving core problems
- Emphasizes the importance of kernel fusion and graph compilation over hand-written kernels

## Main Critiques of Tenstorrent

1. **Over-abstraction**: Too many software layers (TT-Forge, TTorch, TTNN, etc.)
2. **Wrong approach**: Writing individual kernels for each operation instead of using graph compilation
3. **API instability**: C++23 APIs that are constantly changing
4. **Missing the point**: Focusing on mid-tier LLM demos instead of exposing hardware programmability

## Philosophical Commentary

George spends considerable time discussing:
- The meaninglessness of goal-oriented behavior vs. enjoying the journey
- Critique of Silicon Valley's "fake economy" and zero-sum thinking
- The importance of building things that empower individuals vs. large organizations
- Anti-high-frequency-trading rants about value creation vs. extraction

## Advice to Tenstorrent

His core recommendation: **Stop building multiple abstraction layers and focus on a simple, CUDA-like C API that exposes the hardware's programmability advantages.** He argues that Tenstorrent's unique architecture could succeed in new domains, but only if developers can actually access its capabilities without fighting through complex software stacks.

The stream showcases both George's technical expertise in GPU/accelerator architectures and his broader philosophy about building technology that serves users rather than creating artificial complexity.