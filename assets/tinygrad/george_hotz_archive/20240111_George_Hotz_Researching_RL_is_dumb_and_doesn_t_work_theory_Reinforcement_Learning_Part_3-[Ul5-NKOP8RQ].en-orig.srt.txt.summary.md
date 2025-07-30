# George Hotz Stream Summary: Reinforcement Learning Critique

George Hotz discusses his theory on why reinforcement learning (RL) is problematic and ineffective in practical applications. Key points include:

## Problems with Reinforcement Learning
- **Non-stationary data**: Distribution changes over time
- **Feedback loop issues**: The model affects the data which is then used to train the model
- **Credit assignment problems**: Difficulty determining which actions led to rewards
- **Sample inefficiency**: Requires too many interactions to learn effectively
- **Brittle performance**: Small changes can break learned behaviors

## Alternative Approaches
- **Imitation learning**: More stable but can't exceed the performance of what it's imitating
- **Decision transformers**: Treating RL as a sequence prediction problem
- **World models**: Learning predictive models without reconstruction (discussing papers like Dreamer V3, Mew Dreamer)

## Implementation Work
- Shows his implementation attempts with PPO in "tinygrad"
- Struggles with hyperparameter tuning and inconsistent results
- Notes that even fixing bugs sometimes worsens performance

## Future Directions
- Announces a $1,000 bounty for implementing efficient RL algorithms in tinygrad
- Mentions potential researcher position at "tiny Corp" focused on wall training time optimization
- Discusses "tiny boxes" - specialized hardware for AI training
- Expresses interest in solving complex games like Mario 64

## Philosophical Stance
- References "the bitter lesson" - that general methods leveraging computation are ultimately most effective
- Believes RL might work better on high-dimensional problems rather than simplified environments
- Suggests companies may be keeping advances private, though questions if real progress has occurred

The stream concludes with Hotz reaffirming that while RL has theoretical promise, current approaches have significant limitations that make them impractical for many real-world applications.