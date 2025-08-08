## Running machine learning models locally

This is a collection of links and information to assess the viability of
running machine learning models and large language models locally. The aim is
to support engineers and stakeholders to make a well-informed decision when
procuring LLM infrastructure.

The following use cases are considered, based on the [LocalScore](https://www.localscore.ai/about) benchmark:

| Use<br>case | Description                                                                      | Prompt<br>Tokens | Text Generation<br>Tokens |
|-------------|----------------------------------------------------------------------------------|-----------------:|--------------------------:|
| UC1         | Classification, sentiment analysis, keyword extraction                           | 1024             | 16                        |
| UC2         | Long document Q&A, RAG, short summary of extensive text                          | 4096             | 256                       |
| UC3         | Complex reasoning, chain-of-thought, long-form creative writing, code generation | 1280             | 3072                      |
| UC4         | Prompt expansion, explanation generation, creative writing, code generation      | 384              | 1152                      |

A workflow is considered -
-   *interactive*, if input tokens are processed in less than 3 seconds, and the output is produced at more than 10 tokens/s
-   *background task*, if input processing + output generation finishes within 300 seconds (5 minutes)
-   *off-line*, if input processing + output generation finishes within a few hours / overnight (progress indicator is necessary)
-   *non-viable*, if finishing the task takes prohibitively long

We examine four systems: embedded, desktop, workstation, and server:

| AMD CPU example       | CPU type    | Cores | RAM<br>channels | RAM<br>[GB] | RAM WB<br>[GB/s] | Max GPUs   | Max GPU<br>VRAM [GB] | Max ML model<br>size [B params] |
|-----------------------|-------------|------:|----------------:|------------:|-----------------:|-----------:|---------------------:|--------------------------------:|
| Ryzen AI MAX+ PRO 395 | Embedded    | 16    | 2               | 128         | 128              | integrated |                   96 |                              40 |
| Ryzen 9 9950X         | Desktop     | 16    | 2               | 192         | 89.6             |      1 (2) |                   96 |                              40 |
| Threadripper 7970X    | Workstation | 32    | 4               | 256         | 166.4            |      2 (6) |                  192 |                              80 |
| EPYC 9554P            | Server      | 64    | 12              | 384         | 460.8            |      6 (8) |                  576 |                             300 |


The following hardware configurations are examined.

| AMD CPU example       | NVIDIA GPU type                 | GPU VRAM<br>[GB] | ML model size<br>[GB] | Prompt<br>processing<br>[token/s] | Token<br>generation<br>[token/s] | UC1 | UC2 | UC3 | UC4 |
|-----------------------|---------------------------------|-----------------:|----------------------:|----------------------------------:|---------------------------------:|-----|-----|-----|-----|
| Ryzen AI MAX+ PRO 395 |        -                        |                - |                    10 |  84                               | 11                               | ✅ | ❌ | ❌ | ❓ |
| Ryzen 9 9950X         |        -                        |                - |                    10 |  125                              | 8                                | ✅ | ❌ | ❓ | ❓ |
| Ryzen 9 9950X         | [RTX 5080][ls754]               |               16 |                    10 |  2291                             | 24                               | ✅ | ✅ | ❓ | ❓ |
| Ryzen 9 9950X         | [RTX 5090][ls175]               |               32 |                    10 |  4787                             | 65                               | ✅ | ✅ | ✅ | ✅ |
| Threadripper 7970X    |        -                        |                - |                    10 |  223                              | 14                               | ✅ | ❌ | ✅ | ❓ |
| Threadripper 7970X    | [RTX PRO 6000 Blackwell][ls939] |               96 |                    10 |  5126                             | 81                               | ✅ | ✅ | ✅ | ✅ |
| Threadripper 7970X    | 2 x RTX PRO 6000 Blackwell      |              192 |                max 96 |  ?                                | ?                                | ✅ | ✅ | ✅ | ✅ |
| EPYC 9554P            |        -                        |                - |                    10 |  295?                             | 20?                              | ✅ | ❌ | ✅ | ❓ |
| EPYC 9554P            | 4 x RTX PRO 6000 Blackwell      |              384 |               max 192 |  ?                                | ?                                | ✅ | ✅ | ✅ | ✅ |

[ls754]: https://www.localscore.ai/result/754
[ls175]: https://www.localscore.ai/result/175
[ls939]: https://www.localscore.ai/result/939

See the detailed analysis in [local_ml_models.md](local_ml_models.md)
