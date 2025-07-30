# George Hotz Programming Decision Transformer Stream Summary

George Hotz attempts to implement a Decision Transformer for reinforcement learning tasks like LunarLander and Cart Pole using tinygrad. The stream showcases the challenges and frustrations typical in reinforcement learning development.

## Key points:

- **Decision Transformer concept**: Unlike traditional RL methods, Decision Transformers condition on desired future returns to predict actions

- **Implementation challenges**:
  - Tensor dimension mismatches
  - JIT compilation issues in tinygrad
  - Difficulty with causal masking implementation
  - Model initialization problems
  - Hyperparameter tuning struggles

- **Simplified test environment**: Creates "press the light up button" game to debug the model (agent must press whichever button lights up)
  - Successfully trains on simple versions
  - Struggles with increased complexity (more buttons, longer sequences)

- **Results**:
  - Cart Pole: Limited success
  - LunarLander: Repeatedly fails to land properly, often developing "LunarFaller" behavior
  - Simplest game version: Eventually works after significant debugging

- **Conclusions**:
  - George expresses frustration with reinforcement learning: "reinforcement learning doesn't work"
  - Identifies lack of debugging tools as a major challenge
  - Plans to continue working on improvements in future streams

The stream demonstrates the iterative nature of ML development and the particular challenges of reinforcement learning compared to supervised learning approaches.