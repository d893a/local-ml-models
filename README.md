## Running machine learning models locally

This is a collection of links and information to assess the viability of
running machine learning models and large language models locally. The aim is
to support engineers and stakeholders to make a well-informed decision when
procuring LLM infrastructure.

Actual hardware configuration is detailed in
[local_ml_hardware_alternatives.md](local_ml_hardware_alternatives.md#server).

### Use cases

The following use cases are considered, based on the [LocalScore](https://www.localscore.ai/about) benchmark:

| Use<br>case | Description                                                                      | Prompt<br>Tokens | Text Generation<br>Tokens |
|-------------|----------------------------------------------------------------------------------|-----------------:|--------------------------:|
| UC1         | Classification, sentiment analysis, keyword extraction                           | 1024             | 16                        |
| UC2         | Long document Q&A, RAG, short summary of extensive text                          | 4096             | 256                       |
| UC3         | Complex reasoning, chain-of-thought, long-form creative writing, code generation | 1280             | 3072                      |
| UC4         | Prompt expansion, explanation generation, creative writing, code generation      | 384              | 1152                      |

### Levels of interactiveness

A workflow is considered -
-   *Interactive workflow*: Input tokens are processed in less than 5 seconds, and the output is produced at more than 10 tokens/s.
-   *Background task*: Input processing + output generation finishes within 300 seconds (5 minutes).
-   *Off-line workflow*: Input processing + output generation finishes within a few hours / overnight. In this case a progress indicator is helpful.
-   *Non-viable*, if the task takes prohibitively long time to finish.

### Hardware platforms:

We examine four systems: embedded/mobile, desktop, workstation, and server.

| Platform    | AMD CPU example       | Cores | RAM<br>channels | RAM<br>[GB] | RAM WB<br>[GB/s] | Max GPUs   | Max GPU<br>VRAM [GB] | Max ML model<br>size [B params] |
|-------------|-----------------------|------:|----------------:|------------:|-----------------:|-----------:|---------------------:|--------------------------------:|
| Embedded    | Ryzen AI MAX+ PRO 395 | 16    | 2               | 128         | 128              | integrated |                   96 |                              40 |
| Desktop     | Ryzen 9 9950X         | 16    | 2               | 192         | 89.6             |      1 (2) |                   96 |                              40 |
| Workstation | Threadripper 7970X    | 32    | 4               | 256         | 166.4            |      2 (6) |                  192 |                              80 |
| Server      | EPYC 9554P            | 64    | 12              | 384         | 460.8            |      6 (8) |                  576 |                             300 |

<div class="page"/>

The following hardware configurations are examined.

| AMD CPU example       | NVIDIA GPU type                 | GPU VRAM<br>[GB] | ML model size<br>[GB] | Prompt<br>processing<br>[token/s] | Token<br>generation<br>[token/s] | UC1 | UC2 | UC3 | UC4 |
|-----------------------|---------------------------------|-----------------:|----------------------:|----------------------------------:|---------------------------------:|:---:|:---:|:---:|:---:|
| Ryzen AI MAX+ PRO 395 |        -                        |                - |                    10 |  84                               | 11                               |  √  |  -  |  -  |  ?  |
| Ryzen 9 9950X         |        -                        |                - |                    10 |  125                              | 8                                |  √  |  -  |  ?  |  ?  |
| Ryzen 9 9950X         | RTX 5080                        |               16 |                    10 |  [2291][ls754]                    | 24                               |  √  |  √  |  ?  |  ?  |
| Ryzen 9 9950X         | RTX 5090                        |               32 |                    10 |  [4787][ls175]                    | 65                               |  √  |  √  |  √  |  √  |
| Threadripper 7970X    |        -                        |                - |                    10 |  223                              | 14                               |  √  |  -  |  √  |  ?  |
| Threadripper 7970X    | RTX PRO 6000 Blackwell          |               96 |                    10 |  [5126][ls939]                    | 81                               |  √  |  √  |  √  |  √  |
| Threadripper 7970X    | 2 x RTX PRO 6000 Blackwell      |              192 |                max 96 |  ?                                | ?                                |  √  |  √  |  √  |  √  |
| EPYC 9554P            |        -                        |                - |                    10 |  223?                             | 21?                              |  √  |  -  |  √  |  ?  |
| EPYC 9554P            | 4 x RTX PRO 6000 Blackwell      |              384 |               max 192 |  ?                                | ?                                |  √  |  √  |  √  |  √  |

[ls754]: https://www.localscore.ai/result/754
[ls175]: https://www.localscore.ai/result/175
[ls939]: https://www.localscore.ai/result/939
