## Running machine learning models locally

This is a collection of links and information to assess the viability of
running machine learning models and large language models locally. The aim is
to support engineers and stakeholders to make a well-informed decision when
procuring LLM infrastructure.

Actual hardware configuration is detailed in
[local_ml_hardware_alternatives.md](local_ml_hardware_alternatives.md#server).

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
| Ryzen 9 9950X         | RTX 5080                        |               16 |                    10 |  [2291][ls754]                    | 24                               | ✅ | ✅ | ❓ | ❓ |
| Ryzen 9 9950X         | RTX 5090                        |               32 |                    10 |  [4787][ls175]                    | 65                               | ✅ | ✅ | ✅ | ✅ |
| Threadripper 7970X    |        -                        |                - |                    10 |  223                              | 14                               | ✅ | ❌ | ✅ | ❓ |
| Threadripper 7970X    | RTX PRO 6000 Blackwell          |               96 |                    10 |  [5126][ls939]                    | 81                               | ✅ | ✅ | ✅ | ✅ |
| Threadripper 7970X    | 2 x RTX PRO 6000 Blackwell      |              192 |                max 96 |  ?                                | ?                                | ✅ | ✅ | ✅ | ✅ |
| EPYC 9554P            |        -                        |                - |                    10 |  223?                             | 21?                              | ✅ | ❌ | ✅ | ❓ |
| EPYC 9554P            | 4 x RTX PRO 6000 Blackwell      |              384 |               max 192 |  ?                                | ?                                | ✅ | ✅ | ✅ | ✅ |

[ls754]: https://www.localscore.ai/result/754
[ls175]: https://www.localscore.ai/result/175
[ls939]: https://www.localscore.ai/result/939


See the detailed analysis in [local_ml_models.md](local_ml_models.md)


Table fo contents:
- [Introduction](local_ml_models.md#introduction)
- [Hardware](local_ml_models.md#hardware)
    - [CPU only](local_ml_models.md#cpu-only)
        - [Maximum theoretical system memory bandwidth calculation](local_ml_models.md#maximum-theoretical-system-memory-bandwidth-calculation)
    - [Desktop CPUs](local_ml_models.md#desktop-cpus)
    - [Workstation CPUs](local_ml_models.md#workstation-cpus)
    - [Server CPUs](local_ml_models.md#server-cpus)
    - [Mobile CPUs with integrated NPU](local_ml_models.md#mobile-cpus-with-integrated-npu)
- [LLM system characteristics](local_ml_models.md#llm-system-characteristics)
    - [LLM inference performance indicators](local_ml_models.md#llm-inference-performance-indicators)
        - [Prompt Processing Throughput](local_ml_models.md#prompt-processing-throughput)
        - [Time to First Token](local_ml_models.md#time-to-first-token)
        - [Token Generation Throughput](local_ml_models.md#token-generation-throughput)
        - [Model Size](local_ml_models.md#model-size)
        - [Quantization](local_ml_models.md#quantization)
        - [Prompt Length](local_ml_models.md#prompt-length)
        - [Batch Size](local_ml_models.md#batch-size)
        - [CPU Performance](local_ml_models.md#cpu-performance)
            - [Example](local_ml_models.md#example)
        - [Thread/Parallelism Efficiency](local_ml_models.md#threadparallelism-efficiency)
        - [Software Optimization](local_ml_models.md#software-optimization)
        - [GPU parameters](local_ml_models.md#gpu-parameters)
        - [Memory Bandwidth and Latency](local_ml_models.md#memory-bandwidth-and-latency)
        - [Required operations per token](local_ml_models.md#required-operations-per-token)
        - [Hyperthreading: Use one thread per CPU core](local_ml_models.md#hyperthreading-use-one-thread-per-cpu-core)
- [Benchmarks](local_ml_models.md#benchmarks)
    - [Benchmark aggregator sites](local_ml_models.md#benchmark-aggregator-sites)
    - [Benchmarking sites](local_ml_models.md#benchmarking-sites)
- [Performance](local_ml_models.md#performance)
- [Models](local_ml_models.md#models)
- [Benchmark](local_ml_models.md#benchmark)
    - [CPU benchmark](local_ml_models.md#cpu-benchmark)
    - [GPU benchmark](local_ml_models.md#gpu-benchmark)
    - [Model benchmark](local_ml_models.md#model-benchmark)
    - [CPU + NPU benchmark](local_ml_models.md#cpu--npu-benchmark)
- [Hardware](local_ml_models.md#hardware-1)
    - [CPU](local_ml_models.md#cpu)
        - [Embedded CPU](local_ml_models.md#embedded-cpu)
        - [Desktop CPU](local_ml_models.md#desktop-cpu)
        - [Workstation CPU](local_ml_models.md#workstation-cpu)
        - [Server CPU](local_ml_models.md#server-cpu)
    - [CPU cooler](local_ml_models.md#cpu-cooler)
    - [Memory](local_ml_models.md#memory)
- [Performance](local_ml_models.md#performance-1)
    - [Motherboard](local_ml_models.md#motherboard)
    - [SSD](local_ml_models.md#ssd)
    - [PSU](local_ml_models.md#psu)
    - [GPU](local_ml_models.md#gpu)
        - [NVIDIA GPU](local_ml_models.md#nvidia-gpu)
        - [AMD GPU](local_ml_models.md#amd-gpu)
    - [GPU link](local_ml_models.md#gpu-link)
    - [Case](local_ml_models.md#case)
    - [Mini PC](local_ml_models.md#mini-pc)
    - [Mini PC + eGPU](local_ml_models.md#mini-pc--egpu)
    - [tinybox](local_ml_models.md#tinybox)
    - [PIM](local_ml_models.md#pim)
- [Survey](local_ml_models.md#survey)
- [Software](local_ml_models.md#software)
    - [AMD CPU](local_ml_models.md#amd-cpu)
    - [AMD GPU](local_ml_models.md#amd-gpu-1)
    - [NVIDIA GPU](local_ml_models.md#nvidia-gpu-1)
    - [Multi-GPU](local_ml_models.md#multi-gpu)
- [Library](local_ml_models.md#library)
- [Framework](local_ml_models.md#framework)
    - [tinygrad](local_ml_models.md#tinygrad)
    - [TextSynth Server](local_ml_models.md#textsynth-server)
    - [Text](local_ml_models.md#text)
    - [Visual analysis and generation](local_ml_models.md#visual-analysis-and-generation)
    - [Voice](local_ml_models.md#voice)
    - [OCR](local_ml_models.md#ocr)
    - [Embedding](local_ml_models.md#embedding)
- [Papers](local_ml_models.md#papers)

