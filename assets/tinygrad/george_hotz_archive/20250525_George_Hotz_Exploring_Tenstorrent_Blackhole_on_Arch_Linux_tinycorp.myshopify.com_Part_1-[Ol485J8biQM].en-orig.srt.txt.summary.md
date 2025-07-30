# George Hotz Stream Summary: Tenstorrent Blackhole Exploration Part 1

This stream documents George Hotz's unboxing and initial exploration of the Tenstorrent Blackhole AI accelerator card, showcasing both the hardware setup and his critiques of the software ecosystem.

## Key Hardware Setup

**Unboxing Experience:**
- Tenstorrent Blackhole card arrived from Canada in professional packaging
- Card requires PCIe power cables (not included) and draws significant power
- Installation into older desktop computer running Arch Linux
- Immediate noise issues - the card's blower fan is notably loud even at idle

**Technical Specifications Explored:**
- 32GB of high-bandwidth memory
- Multiple PCIe bars for different memory regions (512MB, 1MB, 32GB)
- Risk-V cores integrated on the chip
- Theoretical peak performance capabilities

## Software Challenges

**Installation Difficulties:**
- Tenstorrent officially only supports Ubuntu 22.04, not Arch Linux
- Multiple dependency issues and hardcoded paths in build scripts
- Required specific versions of tools (Clang 17) not available in standard repos
- Eventually resorted to Docker container to get basic functionality

**Driver and API Issues:**
- Complex multi-layered software stack (TT-Metal, TT-Metalium, TTNN)
- C++23 dependencies requiring bleeding-edge compilers
- Numerous abstraction layers making simple tasks difficult
- Poor error messages and documentation gaps

## Technical Deep Dive

**Low-Level Hardware Access:**
- Successfully mapped PCIe memory regions using Python
- Explored memory layout and register mappings
- Investigated firmware loading and Risk-V core access
- Demonstrated direct hardware manipulation bypassing high-level APIs

**Performance Testing:**
- Basic matrix multiplication achieved reasonable performance (~12 teraflops)
- Tensor operations limited to rank-4 tensors only (major limitation)
- Memory bandwidth utilization appeared suboptimal
- Compiler cache requirements for acceptable performance

## Major Criticisms

**Software Architecture:**
- Too many abstraction layers without clear benefit
- Mixing of high-level operations (ReLU) with low-level hardware control
- C++ complexity creating unnecessary barriers to entry
- Poor packaging and dependency management

**Hardware Design Questions:**
- Overly complex architecture with questionable design choices
- Loud thermal management (fan policy issues)
- Power consumption concerns (60+ watts at idle)
- Unclear value proposition of architectural complexity

## Comparisons and Context

**AMD vs Tenstorrent:**
- Praised AMD's strategy of copying NVIDIA's approach
- AMD's recent improvements in software packaging and stability
- Tenstorrent's approach seen as reinventing everything unnecessarily

**Development Philosophy:**
- Advocated for simpler, Python-based driver development
- Emphasized importance of self-serve software that "just works"
- Criticized enterprise-focused complexity over developer experience

## Key Recommendations for Tenstorrent

1. **Fix the fan noise** - critical usability issue
2. **Simplify software stack** - reduce abstraction layers
3. **Focus on documentation** - hardware programming model unclear
4. **Support standard development practices** - move away from C++ complexity
5. **Enable pip-installable drivers** - like Tinygrad's approach

## Overall Assessment

George concludes that while Tenstorrent has shipped impressive hardware, the software ecosystem is "worse than AMD" and creates unnecessary barriers for developers. The card shows potential but needs fundamental rethinking of its software approach to achieve adoption. The stream demonstrates both the technical capabilities and the significant challenges facing alternative AI accelerator companies trying to compete with NVIDIA's established ecosystem.

The documentation improvements (via Corsix's work) were noted as positive developments, but the core software architecture issues remain substantial obstacles to widespread adoption.

---

# George Hotz's Tenstorrent Blackhole Exploration Stream Summary

## Key Points

- George Hotz unboxed and attempted to use a Tenstorrent Blackhole AI accelerator card on an Arch Linux system
- The main issues encountered were:
  - Extremely loud fan noise even when idle (a persistent complaint throughout the stream)
  - Poor software stack with many dependencies and compatibility issues
  - Documentation gaps and confusing abstraction layers
  - Complex installation process that worked only on specific Ubuntu versions

## Technical Journey

- Set up Arch Linux and installed the Tenstorrent drivers
- Struggled with numerous software dependencies and build issues
- Compared the experience to his own Tinygrad project, which he demonstrated working with AMD GPUs with minimal setup
- Eventually got basic matrix multiplication working but couldn't run more complex models
- Created a simple Python script to access the card's memory directly

## Industry Commentary

- Criticized Tenstorrent's software approach as overly complex and impractical
- Compared hardware vendors' approaches (AMD, NVIDIA, Intel, Qualcomm)
- Praised AMD's strategy of compatibility with NVIDIA's ecosystem
- Noted that AI accelerator companies need to focus on developer experience and ease of use
- Discussed how excessive complexity without product-market fit is problematic

## Recommendations for Tenstorrent

- Fix the fan noise/thermal management as highest priority
- Simplify the software stack and improve documentation
- Focus on making simple examples work reliably across different systems
- Reconsider the architecture's complexity and abstraction layers
- Consider implementing a Python-based driver for better accessibility

The stream concluded with Hotz expressing that while the hardware itself might be interesting, the current software implementation makes it impractical for most users.
