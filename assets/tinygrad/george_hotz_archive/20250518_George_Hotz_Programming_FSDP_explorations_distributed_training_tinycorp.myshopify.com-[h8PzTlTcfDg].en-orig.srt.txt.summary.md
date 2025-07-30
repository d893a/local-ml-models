# TinyGrad FSDP Exploration Stream Summary

In this programming stream, George Hotz explored Fully Sharded Data Parallelism (FSDP) for distributed training in TinyGrad, his neural network framework that competes with PyTorch and JAX.

## Key Technical Topics:

1. **Understanding FSDP**
   - Explored how FSDP differs from regular data parallelism by sharding model parameters, gradients, and optimizer states across GPUs
   - Studied how FSDP reduces memory usage per device for training large models

2. **Memory Usage Investigation**
   - Identified inefficient memory usage in current multi-GPU implementation
   - Discovered that simply sharding the data still resulted in duplicated memory across devices
   - Used visualization tools to analyze memory allocation patterns

3. **Optimizer Refactoring**
   - Implemented a "fused atom" optimizer approach that concatenates parameters into a single tensor
   - Demonstrated how this approach simplifies code and improves performance
   - Showed how the fused approach provides a clear path to implementing FSDP

4. **TinyGrad Architecture**
   - Demonstrated TinyGrad's visualization capabilities for neural network computation
   - Explained how TinyGrad generates and optimizes kernels for different hardware
   - Highlighted that TinyGrad is only ~13K lines of code but includes GPU drivers

## Next Steps:
- Refactor all optimizers to use the fused approach
- Fix identified bugs in memory management
- Implement proper sharding of model parameters, gradients and optimizer states
- Improve scheduling of operations across devices

The stream concluded with the realization that there are several prerequisite improvements needed before proper FSDP implementation, but a clear path forward was identified.