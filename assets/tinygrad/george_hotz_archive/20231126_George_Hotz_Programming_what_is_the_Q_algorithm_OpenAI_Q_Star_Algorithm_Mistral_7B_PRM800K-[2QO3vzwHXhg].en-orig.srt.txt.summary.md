# George Hotz Programming Stream Summary

This stream features George Hotz (geohot) exploring the rumored OpenAI "Q*" algorithm (referred to as "qar") while implementing a chat interface for the Open Hermes 2.5 model (a Mistral 7B fine-tune) in his tinygrad framework.

## Main Activities and Accomplishments:

1. **Investigating Q*** - George begins by searching for information about the mysterious Q* algorithm that OpenAI reportedly developed, finding limited concrete information but discovering a paper on "improving mathematical reasoning with process supervision" which might be related.

2. **Model Implementation** - He implements the Open Hermes 2.5 model in tinygrad, working through:
   - Loading model weights
   - Setting up proper model architecture (attention mechanism, feed forward layers)
   - Fixing tokenization issues with special tokens (I'm start, I'm end)
   - Creating a functional chat interface

3. **Tokenizer Challenges** - Significant time was spent modifying the sentence piece tokenizer to properly handle special tokens, requiring protobuf file modifications.

4. **Python Code Execution** - He creates a feature allowing the model to run Python code when prompted, with a human safety check before execution.

5. **Mathematical Testing** - Tests the model with various math problems and finds it surprisingly capable at solving equations, including quadratic formula problems.

6. **Multi-Agent Experiment** - Attempts to create a system where two instances of the model could talk to each other, but runs into issues with the shared KV cache.

7. **Voice Interface Demo** - Briefly demonstrates a separate project that combines TTS, LLM, and speech recognition for a conversational agent named "Stacy."

George was impressed by the capabilities of the 7B parameter model and published his work to a branch called "mistol" in tinygrad. He remains skeptical that Q* represents a major breakthrough, suggesting it might simply be process supervision for mathematical reasoning.