# George Hotz tinygrad Stream Summary

This transcript captures a livestream by George Hotz (geohot) focused on optimizing the ResNet-50 neural network in tinygrad. After some initial banter about his new Meta Quest 3 VR headset, the stream focuses on technical work.

## Key Technical Achievements

- **New Optimization Infrastructure**: George built upon previous work with a more structured approach to optimizing neural network kernels
- **Greedy Search Implementation**: Created a system that tries different optimization actions (upcast, local, etc.) and selects the best performer at each step
- **Significant Performance Gains**:
  - Started at ~393ms execution time for ResNet-50
  - Achieved ~167ms after optimization (roughly 2x speedup)
  - Still only utilizing 26.5% of theoretical GPU maximum (2.7 TFlops vs 10.4 TFlops possible)

## Technical Details

- Built helper functions like `time_linearizer` and `get_linearizer_actions` to make search implementation easier
- Created a caching system to avoid redundant computations
- The greedy approach explores a large but manageable subset of the combinatorial search space (~5.4 million possible permutations)
- Visualization shows different optimization types (yellows for output space, purples for unrolled loops, reds for reduced loops, etc.)

## Future Work

- Replace greedy search with reinforcement learning (RL) approaches
- Implement correctness checking ("fuzzing") for the optimized kernels
- Replace the older optimizer ("kopt") with this new approach
- Try other search methodologies like never-grad

The stream concludes with George raiding another Twitch streamer called LX who is developing a game.