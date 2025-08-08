## Running machine learning models locally

This is a collection of links and information to assess the viability of
running machine learning models and large language models locally. The aim is
to support engineers and stakeholders to make a well-informed decision when
procuring LLM infrastructure.

The following use cases are considered, based on [LocaScore](https://www.localscore.ai/about):

| Use<br>case | Description                                                                      | Prompt<br>Tokens | Text Generation<br>Tokens |
|-------------|----------------------------------------------------------------------------------|-----------------:|--------------------------:|
| UC1         | Classification, sentiment analysis, keyword extraction                           | 1024             | 16                        |
| UC2         | Long document Q&A, RAG, short summary of extensive text                          | 4096             | 256                       |
| UC3         | Complex reasoning, chain-of-thought, long-form creative writing, code generation | 1280             | 3072                      |
| UC4         | Prompt expansion, explanation generation, creative writing, code generation      | 384              | 1152                      |

The following hardware configurations are examined. We expect the large language model to finish processing the input in 10 seconds, and then produce output at a minimum of 10 token/s.

| AMD CPU example       | Cores | RAM<br>channels | RAM<br>[GB] | RAM WB<br>[GB/s] | NVIDIA GPU type                 | GPU VRAM<br>[GB] | ML model size<br>[GB] | Prompt<br>processing<br>[token/s] | Token<br>generation<br>[token/s] | UC1 | UC2 | UC3 | UC4 |
|-----------------------|------:|----------------:|------------:|-----------------:|---------------------------------|-----------------:|----------------------:|----------------------------------:|---------------------------------:|-----|-----|-----|-----|
| Ryzen AI MAX+ PRO 395 | 16    | 2               | 128         | 128              |        -                        |                - |                    10 |  84                               | 11                               | ✅ | ❌ | ❌ | ❓ |
| Ryzen 9 9950X         | 16    | 2               | 192         | 89.6             |        -                        |                - |                    10 |  125                              | 8                                | ✅ | ❌ | ❓ | ❓ |
| Ryzen 9 9950X         | 16    | 2               | 192         | 89.6             | [RTX 5080][ls754]               |               16 |                    10 |  2291                             | 24                               | ✅ | ✅ | ❓ | ❓ |
| Ryzen 9 9950X         | 16    | 2               | 192         | 89.6             | [RTX 5090][ls175]               |               32 |                    10 |  4787                             | 65                               | ✅ | ✅ | ✅ | ✅ |
| Threadripper 7970X    | 32    | 4               | 256         | 166.4            |        -                        |                - |                    10 |  223                              | 14                               | ✅ | ❌ | ✅ | ❓ |
| Threadripper 7970X    | 32    | 4               | 256         | 166.4            | [RTX PRO 6000 Blackwell][ls939] |               96 |                    10 |  5126                             | 81                               | ✅ | ✅ | ✅ | ✅ |
| Threadripper 7970X    | 32    | 4               | 256         | 166.4            | 2 x RTX PRO 6000 Blackwell      |              192 |                max 96 |  ?                                | ?                                | ✅ | ✅ | ✅ | ✅ |
| EPYC 9554P            | 64    | 12              | 384         | 460.8            |        -                        |                - |                    10 |  295?                             | 20?                              | ✅ | ❌ | ✅ | ❓ |
| EPYC 9554P            | 64    | 12              | 768         | 460.8            | 4 x RTX PRO 6000 Blackwell      |              384 |               max 192 |  ?                                | ?                                | ✅ | ✅ | ✅ | ✅ |

A workflow is considered -
-   *interactive*, if input processing takes less than 3 seconds, and the output it produced at more than 10 tokens/s
-   *background task*, if input processing + output generation finishes within 300 seconds (5 minutes)
-   *off-line*, if input processing + output generation finishes within a few hours (progress indicator is necessary)
-   *non-viable*, if takes prohibitively long

[ls754]: https://www.localscore.ai/result/754
[ls175]: https://www.localscore.ai/result/175
[ls939]: https://www.localscore.ai/result/939

Running machine learning models requires:
- Hardware
- Models
- Software

## Hardware

LLM inference prefill (preprocessing) is compute-bound: The number of
processing cores matter most. In the case of CPU inference CPU clock frequency,
then (L3 and L1) cache size. Token generation is memory bandwidth-bound:
Higher memory throughput results in faster token generation.

GPUs have both high computing power and high memory bandwidth. A GPU with even
a small amount of memory can accelerate inference speeds considerably.

The system RAM should be large enough to accommodate the model plus the
context. If the system RAM is too small, then swapping to (relatively slow)
SSD will degrade overall performance.

In this assessment only AMD CPUs are examined.

The following hardware configurations are considered:
-   CPU only
-   CPU + GPU: The model plus context fits into the GPU VRAM
-   Hybrid CPU + GPU, including mobile (laptop, mini PC) solutions. The model plus
    context fits into the CPU memory, but does not fit into the GPU VRAM.


### CPU only

For LLM inference the CPU needs to support the AVX-512 instruction set's
bfloat16 format. It is available in the AMD [Zen 4](https://en.wikipedia.org/wiki/Zen_4)
and [Zen 5](https://en.wikipedia.org/wiki/Zen_5) architectures.

#### Maximum theoretical system memory bandwidth calculation

The maximum theoretical system memory bandwidth is determined by the memory
modules' speed and how many there are, and how fast the CPU can exchange data
with the memory modules.

$$
BW_{RAM} = RAM channels * 8 * RAM speed [GT/s] \\
BW_{CPU} = n_CCDs per core * 32 * FCLK [GHz] \\
BW = min(BW_{RAM}, BW_{CPU})
$$

Where:
-   $BW_{RAM}$: Theoretical bandwidth on the RAM modules' side
-   $BW_{CPU}$: Theoretical bandwidth on the CPU memory controller side
-   BW: Theoretical system RAM bandwidth.
    -   Actual values are measured between 30% to 95% of the theoretical maximum.
-   FCLK: Fabric Clock Speed - clock speed of the memory controller in the CPU.
    -   Zen 4: 1.8 Ghz, Zen 5: 2.0 GHz. Some models can be overclocked.
-   CCD: Core complex Die - contains the memory controller on the CPU side.
    -   Ranges from 1 to 16 per CPU core.
-   RAM speed: 4.8 GT/s to 8 GT/s.
-   RAM channels: 2-12 (24 for 2-CPU setups).
    -   RAM sizes vary between 4-128 GB
    -   Viable total memory size: 8 GB to 1536 GB. (3072 GB in the case of 24 x 128 GB modules.)
-   See also: https://www.reddit.com/r/threadripper/comments/1azmkvg/comparing_threadripper_7000_memory_bandwidth_for/

Examples:
-   AMD Ryzen 5 7400F: n_CCD = 1, FCLK = 1.8 GHz, ch = 2.
    -   BW = min(2 ch * 8 * 5.2 GT/s, 1 CCD * 32 * 1.8 GHz) = min(83.2, 57.6) GB/s = 57.6 GB/s
    -   If the model size plus context is 10 GB, then the generation throughput is less than 5.8 token/s.
-   AMD EPYC 9755: n_CCD = 16, FCLK = 2.0 GHz, ch = 12.
    -   BW = min(12 ch * 8 * 5.6 GT/s, 16 CCD * 32 * 2.0 GHz) = min(537.6, 1024) GB/s = 537.6 GB/s
    -   If the LLM size plus context is 10 GB, then the generation throughput is less than 53.8 token/s.

In the next sections the following CPU categories are detailed:
-   Desktop CPUs
-   Workstation CPUs
-   Server CPUs
-   Mobile CPUs


### Desktop CPUs

-   Socket: [AM5](https://en.wikipedia.org/wiki/Socket_AM5) (dual-channel RAM)
-   Maximum RAM: 192 GB (limits LLM model + context size to ~100 GB)
-   Maximum cores: 16 (determines prefill throughput)
-   Maximum L3 cache: 32-128 MB (limits prefill throughput)
    -   L1 cache (Zen 4/4c): 32+32 kB, L2: 1 MB, L3: 32 MB per CCD.
    -   L1 cache (Zen 5/5c): 32+48 kB, L2: 1 MB, L3: 32 MB per CCD.
-   Maximum available PCIe lanes: 24.
    -   Enough to handle only one CPIe 5.0 x16 GPU
-   Maximum theoretical memory bandwidth: 89.6 GB/s
    -   This caps token generation for a 10 GB LLM model + context at 9 token/s.

| CPU Series (AM5 socket)          | Cores | Max<br>RAM size | Max<br>RAM BW | Token<br>generation |
|----------------------------------|-------|----------------:|--------------:|--------------------:|
| Ryzen 7700/7900/7950             | 6-16  | 128 GB          | 83.2 GB/s     |         3-8 token/s |
| Ryzen 7040/7045/8000F/8000G      | 6-16  | 128 GB          | 83.2 GB/s     |         3-8 token/s |
| [EPYC 4004][4004]                | 4-16  | 128 GB          | 83.2 GB/s     |         3-8 token/s |
| [Ryzen 9000][9000]               | 6-16  | 192 GB          | 89.6 GB/s     |         3-8 token/s |

[4004]: https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)
[9000]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Granite_Ridge_(9000_series,_Zen_5_based)

Prices for a complete desktop computer are in the 1000--5000 EUR range.

### Workstation CPUs

This includes the AMD Ryzen Threadripper and EPYC 8004 processors.

-   Sockets:
    -   [sTR5](https://en.wikipedia.org/wiki/Socket_sTR5):
        -   Threadripper: 4-channel DDR5-5200/6400 (Zen 4/5)
        -   Threadripper Pro: 8-channel DDR5-5200/6400 (Zen 4/5)
    -   [SP6](https://en.wikipedia.org/wiki/Socket_SP6):
        -   EPYC 8004: 6-channel DDR5-4800 (Zen 4 only)
-   Maximum RAM: 1024 GB (2048 GB?)
-   Maximum cores: 96 (counts at prefill throughput)
-   Cache:
    -   L1: 80 KB (48 KB data + 32 KB instruction) per core.
    -   L2: 1 MB per core
    -   L3: 64-384 MB per CPU
-   Maximum PCIe lanes: 128 (Pro only;
    -   Threadripper: 48 PCIe 5.0 and 24 PCIe 4.0
        -   Enough to add 2 CPIe 5.0 x16 GPUs
            ([TechPowerUp](https://www.techpowerup.com/cpu-specs/ryzen-threadripper-9980x.c4169#gallery-2))
    -   Threadripper Pro: 128 PCIe 5.0 lanes
        -   Enough to add 6 CPIe 5.0 x16 GPUs
            ([TechPowerUp](https://www.techpowerup.com/cpu-specs/ryzen-threadripper-pro-9995wx.c4163#gallery-2))
-   Maximum number of CCDs per CPU: 12
-   Maximum theoretical memory bandwidth: 166.4 to 409.6 GB/s
    -   Caps token generation throughput of 10 GB LLM model at 16 to 40 token/s

| CPU Series               | Socket | Cores   | Max<br>RAM size | Max<br>RAM BW | Token<br>generation |
|--------------------------|--------|--------:|----------------:|--------------:|--------------------:|
| Threadripper 7000X       | sTR5   | 24-64   | 512 GB          | 166.4 GB/s    |        5-16 token/s |
| Threadripper Pro 7000WX  | sTR5   | 12-96   | 1024 GB         | 332.8 GB/s    |       10-33 token/s |
| Threadripper 9000X       | sTR5   | 24-64   | 512 GB          | 204.8 GB/s    |        6-20 token/s |
| Threadripper Pro 9000WX  | sTR5   | 12-96   | 2048 GB         | 409.6 GB/s    |       13-40 token/s |
| EPYC 8004                | SP6    | 8-64    | 768 GB          | 230.4 GB/s    |        7-23 token/s |

Prices for a complete workstation computer are in the 10,000--20,000 EUR range.


### Server CPUs

-   Socket: [SP5](https://en.wikipedia.org/wiki/Socket_SP5):
    -   [EPYC 9004][9004]: 12-channel DDR5-4800 (Zen 4) (24 for 2-CPU config)
    -   [EPYC 9005][9005]: 12-channel DDR5-5600 (Zen 5) (24 for 2-CPU config)
-   Maximum RAM: 3 TB (6 TB for 2-CPU config)
-   Maximum cores: 192 (counts at prefill throughput)
-   Cache:
    -   L1: 80 KB (48 KB data + 32 KB instruction) per core.
    -   L2: 1 MB per core
    -   L3: 16-1152 MB per CPU
-   Maximum PCIe lanes: 128 (160 in 2-CPU config)
    -   Enough to add 8 CPIe 5.0 x16 GPUs
        ([TechPowerUp](https://www.techpowerup.com/cpu-specs/epyc-9755.c3881#gallery-5))
-   Maximum number of CCDs per CPU: 16
    -   Minimum 9 CCDs are required to serve the RAM module bandwidth
        -   Assert $9~CCD \times 32 \times 2.0~GHz \ge 12~ch \times 8 \times 6.0~GT/s$,
            see [calculation](#maximum-theoretical-system-memory-bandwidth-calculation)
-   Maximum theoretical memory bandwidth: 460.8 to 1075.2 GB/s
    -   Caps token generation throughput of 10 GB LLM model at 46 to 100 token/s
    -   Note that from the Genoa platform on, [single-rank memory modules will perform
        well](https://semianalysis.com/2022/11/10/amd-genoa-detailed-architecture-makes/)
        > The other important feature is dual rank versus single rank memory.
        > With Milan and most Intel platforms, dual-rank memory is crucial to
        > maximizing performance. There’s a 25% performance delta on Milan,
        > for example. With Genoa, this is brought down to 4.5%. This is
        > another considerable cost improvement because cheaper single-rank
        > memory can be used.
        ([Slide](https://i0.wp.com/semianalysis.com/wp-content/uploads/2024/11/https3A2F2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com2Fpublic2Fimages2F8aba2a1b-dc51-41c5-a618-3ad93dfcd169_5278x2891-scaled.jpg?ssl=1))


| Series<br>(SP5 socket) | Cores  | Max RAM size<br>(1/2-CPU) | Max RAM BW<br>(1/2-CPU) | Token generation<br>(1/2-CPU) |
|------------------------|-------:|--------------------------:|------------------------:|------------------------------:|
| [EPYC 9004][9004]      | 16-128 | 3 / 6 TB                  | 460.8 / 921.6 GB/s      | 15-46 / 21-72 token/s         |
| [EPYC 9005][9005]      | 6-192  | 3 / 6 TB                  | 537.6 / 1075.2 GB/s     | 18-53 / 25-86 token/s         |

[9004]: https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)
[9005]: https://en.wikipedia.org/wiki/Epyc#Fifth_generation_Epyc_(Grado,_Turin_and_Turin_Dense)

Test results at [OpenBenchmarking.org](https://openbenchmarking.org/) for
[llama.cpp](https://openbenchmarking.org/test/pts/llama-cpp&eval=528957f347896758ab932a93b883fc633206e394#metrics) and
[LocalScore](https://openbenchmarking.org/test/pts/localscore&eval=b2ce18055004793cb1bdfa1d826b3ab6666d1756#metrics).

Prices for a complete computer with SP5 CPU socket are in the 2,500--7,000 EUR range.

###  Mobile CPUs with integrated NPU

-   Sockets: FL1, FP7, FP7r2 or FP8 type packages
    -   200: All models support DDR5-5600 or LPDDR5X-7500 in 128-bit "dual-channel" mode.
    -   300: All models support DDR5-5600 or LPDDR5X-8000 in dual-channel mode.
-   Maximum RAM: 128 GB
-   Maximum cores: 4-12 (counts at prefill throughput)
-   Cache:
    -   L1: 80 KB (48 KB data + 32 KB instruction) per core.
    -   L2: 1 MB per core
    -   L3: 8-64 MB
-   Maximum PCIe lanes: 16-20
-   Maximum theoretical memory bandwidth: 128 GB/s
    -   This caps token generation throughput of a 10 GB LLM model at 12 token/s


| CPU Series (embedded)    | Cores<br>type  | Max RAM    | Max RAM BW   | Token<br>generation |
|--------------------------|---------------:|-----------:|-------------:|--------------------:|
| [Ryzen 8040][8040]       | 4-8            | 128 GB     | 89.6 GB/s    |         3-9 token/s |
| [Ryzen AI 200][200]      | 4-8            | 256 GB     | 128 GB/s     |        4-12 token/s |
| [Ryzen AI 300][300]      | 4-12           | 256 GB     | 128 GB/s     |        4-12 token/s |
| [Ryzen AI MAX/MAX+][MAX] | 6-16           | 128 GB     | 128 GB/s     |        4-12 token/s |

[8040]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Hawk_Point_(8040_series,_Zen_4/RDNA3/XDNA_based)
[200]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Hawk_Point_Refresh_(200_series,_Zen_4/RDNA3/XDNA_based)
[300]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Strix_Point_and_Krackan_Point_(Zen_5/RDNA3.5/XDNA2_based)
[MAX]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Strix_Halo_(Zen_5/RDNA3.5/XDNA2_based)

Prices for a complete computer are in the 1,000--2,500 EUR range.


### CPU + GPU

|                        | **Desktop CPU+<br>Consumer GPU**      | **Workstation<br>GPU**      | **Server<br>GPU**      |
|------------------------|-----------------------------|------------------------|------------------------|
| **CPU [AMD]**          | Threadripper /<br> Threadripper Pro / EPYC | EPYC 8004/9005         | N/A                      |
| **Cores**              | 16-64                       | 64-128                 | 4-16                      |
| **FCLK**               | 16-64                       | 64-128                 | 4-16                      |
| **Number of CCDs**     | 16-64                       | 64-128                 | 4-16                      |
| **L1 cache [kB / core]** | 64-128 MB                   | 128-256 MB             | 8-32 MB                   |
| **L3 cache [MB / CPU]** | 64-128 MB                   | 128-256 MB             | 8-32 MB                   |
| **PCIe lanes**         | 64-128                      | 128-256                | 4-16                       |
| **Number of CPUs**     | 1                            | 128-256                | 4-16                       |
|                        |                             |                        |                           |
| **RAM**                | 64-256 GB                   | 256-1024 GB            | 4-32 GB                    |
| **RAM type**           | GDDR6                       | GDDR6                  | LPDDR4/<br>LPDDR5               |
| **RAM channels**       | 2-4                         | 4-8                    | 1-2                         |
| **RAM speed [MT/s]**   | GDDR6                       | GDDR6                  | LPDDR4/<br>LPDDR5               |
| **RAM Bandwidth**      | ~100 GB/s                   | ~200 GB/s              | ~10-50 GB/s                 |
|                        |                             |                        |                           |
| **Number of GPUs**     | 1-2                         | 1-8                    | 1-4                         |
| **VRAM per GPU**       | 16-48 GB                    | 32-96 GB               | 2-16 GB                     |
| **Total VRAM**         | 16-96 GB                    | 32-768 GB              | 2-64 GB                     |

---

### Hybrid CPU + GPU (Embedded)

|                        | **Mobile CPU + NPU** |
|------------------------|------------------------|
| **CPU [AMD]**          | Ryzen<br>AI 5 330 --<br>AI MAX+ 395 |
| **Cores**              | 4-16                   |
| **FCLK**               | 4-16                   |
| **Number of CCDs**     | 4-16                   |
| **L1 cache [kB / core]** | 16-64 MB             |
| **L3 cache [MB / CPU]** | 16-64 MB              |
| **PCIe lanes**         | 16-64                  |
| **Number of CPUs**     | 1                      |
|                        |                        |
| **RAM**                | 16-64 GB               |
| **RAM type**           | GDDR6                  |
| **RAM channels**       | 1-2                    |
| **RAM speed [MT/s]**   | GDDR6                  |
| **RAM Bandwidth**      | ~50 GB/s               |
|                        |                        |
| **Number of GPUs**     | 1                      |
| **VRAM per GPU**       | 8-24 GB                |
| **Total VRAM**         | 8-24 GB                |
