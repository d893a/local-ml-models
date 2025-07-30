# George Hotz's RL Programming Stream Summary

George Hotz spends a stream exploring why "Reinforcement Learning is dumb and doesn't work" while attempting to get various RL algorithms working on simple environments.

## Key Technical Elements:

- **Test Environments**: Works with a simple "press the light up button" game he created and the LunarLander environment
- **Algorithms Tested**:
  - Vanilla Policy Gradient (VPG)
  - Proximal Policy Optimization (PPO)
  - Advantage Actor-Critic (A2C)
  - Attempted to use Decision Transformers (not successful)
- **Framework Comparison**: Implements solutions in both tinygrad (his framework) and PyTorch, finding PyTorch much faster for small networks
- **Bug Discovery**: Fixes several critical bugs in his implementations but still struggles with reliability

## Key Findings:

- Switching from tanh to ReLU activation significantly improved training stability
- Network architecture matters - adding an extra layer sometimes helped performance
- PPO implementation had bugs but even after fixing them, reliability issues remained
- Extensive hyperparameter tuning was required to get acceptable results
- Eventually achieved successful lunar landings but with inconsistent performance

## Conclusions:

George concludes that RL requires lowering expectations:
- It's unreliable and highly sensitive to initialization and hyperparameters
- It can improve performance on tasks given enough compute, but won't work reliably
- Extensive feature engineering and reward shaping defeats the purpose of end-to-end RL

As he puts it: "If you think you're going to make it work reliably, your expectations are too high. If you think that sometime it might get slightly better at a task after you pour an absurd amount of compute into it, that is the right expectation for RL."

The stream also included side discussions about Twitch's platform policies and potential alternatives like Kick.