## Introduction

This is a collection of links and information to assess the viability of running machine learning models and large language models locally.

The aim is to support engineers and stakeholders to make a well-informed decision when procuring LLM infrastructure.

Running machine learning models requires:
- Hardware
- Models
- Software

## Hardware

LLM inference prefill (preprocessing) is compute-bound: The number of
processing cores matter most. In case of CPU inference CPU clock frequency,
then the L3 and L1 cache size. Token generation is memory bandwidth-bound:
Higher memory throughput results in faster token generation.

GPUs have both high computing power and high memory bandwidth. A GPU with even
a small amount of memory can accelerate inference speeds considerably.

The system RAM should be large enough to accommodate the model plus the
context. If the system RAM is insufficient, then swapping to (relatively slow)
SSD will degrade overall performance.

The following hardware configurations are considered:
-   CPU only
-   CPU + GPU: The model plus context fits into the GPU VRAM
-   Hybrid CPU + GPU, including mobile (laptop, mini PC) solutions. The model plus
    context fits into the CPU memory, but does not fit into the GPU VRAM.


### CPU only

In this assessment only AMD CPUs are examined.

For LLM inference the CPU needs to support the AVX-512 instruction set's
bfloat16 format. It is available in the [Zen 4](https://en.wikipedia.org/wiki/Zen_4)
and [Zen 5](https://en.wikipedia.org/wiki/Zen_5) architectures.

**Maximum theoretical memory bandwidth calculation**

> BW = min(RAM channels * 8 * RAM speed [GT/s], n_CCDs per core * 32 * FCLK [GHz]) where FCLK is the fabric clock speed.

Where:
-   BW: Theoretical RAM Bandwidth. Actual values are measured between 30 to 95% of the theoretical maximum.
-   FCLK: Fabric Clock Speed - clock speed of the memory controller in the
    CPU. Zen 4: 1.8 Ghz, Zen 5: 2.0 GHz. Some models can be overclocked.
-   CCD: Core complex Die - contains the memory controller in the CPU. Ranges from 1 to 16 per CPU core.
-   RAM speed: 4.8 GT/s to 8 GT/s.
-   RAM channels: 2-12 (24 for 2-CPU setups). RAM sizes vary between 4-128 GB
    modules, resulting in total memory size of 8 GB to 3072 GB.
-   See also: https://www.reddit.com/r/threadripper/comments/1azmkvg/comparing_threadripper_7000_memory_bandwidth_for/

Examples:
-   AMD Ryzen 5 7400F: n_CCD = 1, FCLK = 1.8 GHz, ch = 2.
    -   BW = min(2 ch * 8 * 5.2 GT/s, 1 CCD * 32 * 1.8 GHz) = min(83.2, 57.6) GB/s = 57.6 GB/s
    -   If the model size plus context is 10 GB, then the generation throughput is less than 5.8 token/s.
-   AMD EPYC 9755: n_CCD = 16, FCLK = 2.0 GHz, ch = 12.
    -   BW = min(12 ch * 8 * 5.6 GT/s, 16 CCD * 32 * 2.0 GHz) = min(537.6, 1024) GB/s = 537.6 GB/s
    -   If the model size plus context is 10 GB, then the generation throughput is less than 53.8 token/s.

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
-   Maximum available PCIe lanes: 24 (enough to handle only one CPIe 5.0 x16 GPU)
-   Maximum theoretical memory bandwidth: 89.6 GB/s
    -   This caps token generation for a 10 GB LLM model at 9 token/s.

| CPU Series                       | Cores | Max RAM | Max RAM BW |
|----------------------------------|-------|---------|------------|
| Ryzen 7700/7900/7950             | 6-16  | 128 GB  | 83.2 GB/s  |
| Ryzen 7040/7045/8000F/8000G      | 6-16  | 128 GB  | 83.2 GB/s  |
| [EPYC 4004][4004]                | 4-16  | 128 GB  | 83.2 GB/s  |
| [Ryzen 9000][9000]               | 6-16  | 192 GB  | 89.6 GB/s  |

[4004]: https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)
[9000]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Granite_Ridge_(9000_series,_Zen_5_based)

Prices for a complete computer are in the 1000--5000 EUR range.

### Workstation CPUs

This includes the Ryzen Threadripper and EPYC 8004 processors.

-   Sockets:
    -   [sTR5](https://en.wikipedia.org/wiki/Socket_sTR5):
        -   Threadripper: quad-channel DDR5-5200/6400 (Zen 4/5)
        -   Threadripper Pro: octa-channel DDR5-5200/6400 (Zen 4/5)
    -   [SP6](https://en.wikipedia.org/wiki/Socket_SP6):
        -   EPYC 8004: hexa-channel DDR5-4800 (Zen 4 only)
-   Maximum RAM: 1024 GB (2048 GB?)
-   Maximum cores: 96 (counts at prefill throughput)
-   Cache:
    -   L1: 80 KB (48 KB data + 32 KB instruction) per core.
    -   L2: 1 MB per core
    -   L3: 64-384 MB per CPU
-   Maximum PCIe lanes: 128 (Pro only; enough to add 6 CPIe 5.0 x16 GPUs)
-   Maximum number of CCDs per CPU: 12
-   Maximum theoretical memory bandwidth: 166.4 to 409.6 GB/s
    -   Caps token generation throughput of 10 GB LLM model at 16 to 40 token/s

| CPU Series               | Cores   | Max RAM   | Max RAM BW   |
|--------------------------|--------:|----------:|-------------:|
| Threadripper 7000X       | 24-64   | 512 GB    | 166.4 GB/s   |
| Threadripper Pro 7000WX  | 12-96   | 1024 GB   | 332.8 GB/s   |
| EPYC 8004                | 8-64    | 768 GB    | 230.4 GB/s   |
| Threadripper 9000X       | 24-64   | 512 GB    | 409.6 GB/s   |
| Threadripper Pro 9000WX  | 12-96   | 2048 GB   | 204.8 GB/s   |

Prices for a complete computer are in the 10,000--20,000 EUR range.


### Server CPUs

-   Socket:[SP5](https://en.wikipedia.org/wiki/Socket_SP5):
    -   EPYC 9004: 12-channel DDR5-4800 (Zen 4) (24 for 2-CPU config)
    -   EPYC 9005: 12-channel DDR5-5600 (Zen 5) (24 for 2-CPU config)
-   Maximum RAM: 3 TB (6 TB for 2-CPU config)
-   Maximum cores: 192 (counts at prefill throughput)
-   Cache:
    -   L1: 80 KB (48 KB data + 32 KB instruction) per core.
    -   L2: 1 MB per core
    -   L3: 16-1152 MB per CPU
-   Maximum PCIe lanes: 128 (160 in 2-CPU config; enough to add 8 CPIe 5.0 x16 GPUs)
-   Maximum number of CCDs per CPU: 16 (minimum 8 CCD is required to serve RAM BW)
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

        See [Slide](https://i0.wp.com/semianalysis.com/wp-content/uploads/2024/11/https3A2F2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com2Fpublic2Fimages2F8aba2a1b-dc51-41c5-a618-3ad93dfcd169_5278x2891-scaled.jpg?ssl=1)

| Series                | Cores         | Max RAM<br>(1/2-CPU) | Max RAM BW<br>(1/2-CPU) |
|-----------------------|--------------:|---------------------:|------------------------:|
| [EPYC 9004][9004]     | 16-128        | 3 / 6 TB             | 460.8 / 921.6 GB/s      |
| [EPYC 9005][9005]     | 6-192         | 3 / 6 TB             | 537.6 / 1075.2 GB/s     |

[9004]: https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)
[9005]: https://en.wikipedia.org/wiki/Epyc#Fifth_generation_Epyc_(Grado,_Turin_and_Turin_Dense)

Prices for a complete computer are in the 2,500--7,000 EUR range.


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
    -   Caps topen generation throughput of a 10 GB LLM model at 12 token/s


| Series                   | Cores  | Max RAM    | Max RAM BW   |
|--------------------------|-------:|-----------:|-------------:|
| [Ryzen 8040][8040]       | 4-8    | 128 GB     | 89.6 GB/s    |
| [Ryzen AI 200][200]      | 4-8    | 256 GB     | 128 GB/s     |
| [Ryzen AI 300][300]      | 4-12   | 256 GB     | 128 GB/s     |
| [Ryzen AI MAX/MAX+][MAX] | 6-16   | 128 GB     | 128 GB/s     |

[8040]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Hawk_Point_(8040_series,_Zen_4/RDNA3/XDNA_based)
[200]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Hawk_Point_Refresh_(200_series,_Zen_4/RDNA3/XDNA_based)
[300]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Strix_Point_and_Krackan_Point_(Zen_5/RDNA3.5/XDNA2_based)
[MAX]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Strix_Halo_(Zen_5/RDNA3.5/XDNA2_based)

Prices for a complete computer are in the 1000--2,500 EUR range.


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
