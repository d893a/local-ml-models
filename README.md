## Introduction

This is a collection of links and information to assess the viability of running machine learning models and large language models locally.

The aim is to support engineers and stakeholders to make a well-informed decision when procuring LLM infrastructure.

Running machine learning models requires:
- Hardware
- Models
- Software

## Hardware

The following hardware configurations are considered:
-   CPU only
-   CPU + GPU (The model plus context fits into the GPU VRAM)
-   Hybrid CPU + GPU, including embedded (Mini PC) solutions. The model plus
    context fits into the CPU memory, but does not fit into the GPU VRAM.


### CPU only

In CPU inference prefill (preprocessing) is compute-bound: The number or cores
matter most, then CPU clock frequency, then the L3 and L1 cache size. Token
generation is memory bandwidth-bound: higher memory throughput results in
faster token generation. The RAM should be large enough to accommodate the
model plus the context.

In this assessment only AMD CPUs are examined.

For LLM inference the CPU needs to support the AVX-512 instruction set's
bfloat16 format. It is available in the [Zen 4](https://en.wikipedia.org/wiki/Zen_4)
and [Zen 5](https://en.wikipedia.org/wiki/Zen_5) architectures.

**Maximum theoretical memory bandwidth calculation**

> BW = min(RAM channels * 8 * RAM speed [GT/s], n_CCDs per core * 32 * FCLK [GHz]) where FCLK is the fabric clock speed.

Where:
-   BW: Theoretical RAM Bandwidth. Actual values are measured between 30 to 95% of the theoretical maximum.
-   FCLK: Fabric clock speed - clock speed on the CPU side. Zen 4: 1.8 Ghz, Zen 5: 2.0 GHz.
-   CCD: Core complex Die - contains the memory controller in the CPU. Ranges from 1 to 16 per CPU core.
-   RAM speed: 4.8 GT/s to 8 GT/s.
-   RAM channels: 2-12 (24 for 2-CPU setups). RAM sizes vary between 4-128 GB
    modules, resulting in total memory size of 8 GB to 3072 GB.

Examples:
-   AMD Ryzen 5 7400F: n_CCD = 1, FCLK = 1.8 GHz, ch = 2.
    -   BW = min(2 ch * 8 * 5.2 GT/s, 1 CCD * 32 * 1.8 GHz) = min(83.2, 57.6) GB/s = 57.6 GB/s
    -   If the model size plus context is 10 GB, then the generation throughput is less than 5.8 token/s.
-   AMD EPYC 9755: n_CCD = 16, FCLK = 2.0 GHz, ch = 12.
    -   BW = min(12 ch * 8 * 5.6 GT/s, 16 CCD * 32 * 2.0 GHz) = min(537.6, 1024) GB/s = 537.6 GB/s
    -   If the model size plus context is 10 GB, then the generation throughput is less than 53.8 token/s.

The following CPU categories are considered:
-   Desktop CPUs
-   Workstation CPUs
-   Server CPUs
-   Mobile CPUs


|                        | **Consumer<br>CPU only**     | **Workstation<br>CPU only**      | **Server<br>CPU only**      |
|------------------------|------------------------------|----------------------------------|-----------------------------|
| **CPU [AMD]**          | Ryzen                        | Threadripper /<br>Threadripper Pro | EPYC                      |
| **Cores**              | 4-16                         | 16-64                            | 64-128                      |
| **FCLK**               | 4-16                         | 16-64                            | 64-128                      |
| **Number of CCDs**     | 4-16                         | 16-64                            | 64-128                      |
| **L1 cache [MB]**      | 16-64                        | 64-128                           | 128-256                     |
| **L3 cache [MB]**      | 16-64                        | 64-128                           | 128-256                     |
| **PCIe lanes**         | 16-64                        | 64-128                           | 128-256                     |
| **Number of CPUs**     | 1                            | 1                                | 1-2                         |
|                        |                              |                                  |                             |
| **RAM**                | 16-64 GB                     | 64-256 GB                        | 256-1024 GB                 |
| **RAM type**           | DDR4                         | DDR4                             | DDR4                        |
| **RAM channels**       | 1-2                          | 2-4                              | 4-8                         |
| **RAM speed [MT/s]**   | DDR4                         | DDR4                             | DDR4                        |
| **RAM Bandwidth**      | ~50 GB/s                     | ~100 GB/s                        | ~200 GB/s                   |
|                        |                              |                                  |                             |
| **Number of GPUs**     | 0                            | 0                                | 0                           |
| **VRAM per GPU**       | 0                            | 0                                | 0                           |
| **Total VRAM**         | 0                            | 0                                | 0                           |





### Desktop CPUs

-   Socket: [AM5](https://en.wikipedia.org/wiki/Socket_AM5) (dual-channel RAM)
-   Maximum RAM: 192 GB (limits LLM model + context size to ~100 GB)
-   Maximum cores: 16 (limits prefill throughput)
-   Maximum L3 cache: 32-128 MB (limits prefill throughput)
-   Maximum available PCIe lanes: 24 (enough to add only one CPIe 5.0 x16 GPU)
-   Maximum theoretical memory bandwidth: 89.6 GB/s (caps 10 GB LLM model at 9 token/s generation)

- L1 cache (Zen 4/4c): 32+32 kB, L2: 1 MB, L3: 32 MB per CCD.
- L1 cache (Zen 5/5c): 32+48 kB, L2: 1 MB, L3: 32 MB per CCD.

| Series                | Cores      | Max RAM  | Max RAM BW |
|-----------------------|------------|----------|------------|
| Ryzen 7700/7900/7950  | 6-16       | 128 GB   | 83.2 GB/s  |
| Ryzen 7040/7045/8000F/8000G | 6-16 | 128 GB   | 83.2 GB/s  |
| [EPYC 4004][4004]             | 4-16       | 128 GB   | 83.2 GB/s  |
| [Ryzen 9000][9000]            | 6-16       | 192 GB   | 89.6 GB/s  |

[4004]: https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)
[9000]: https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Granite_Ridge_(9000_series,_Zen_5_based)


### Workstation CPUs

This includes the Ryzen Threadripper and EPYC 8004 processors.

-   Sockets:
    -   [sTR5](https://en.wikipedia.org/wiki/Socket_sTR5):
        -   Threadripper: quad-channel DDR5-5200/6400 (Zen 4/5)
        -   Threadripper Pro: octa-channel DDR5-5200/6400 (Zen 4/5)
    -   [SP6](https://en.wikipedia.org/wiki/Socket_SP6):
        -   EPYC 8004: hexa-channel DDR5-4800 (Zen 4 only)
-   Maximum RAM: 1024 GB (limits LLM model + context size to ~500 GB)
-   Maximum cores: 96 (counts at prefill throughput)
-   Cache:
    -   L1: 80 KB (48 KB data + 32 KB instruction) per core.
    -   L2: 1 MB per core
    -   L3: 64-384 MB per CPU
-   Maximum PCIe lanes: 128 (enough to add 6 CPIe 5.0 x16 GPUs)
-   Maximum number of CCDs per CPU: 12
-   Maximum theoretical memory bandwidth: 409.6 GB/s (caps 10 GB LLM model at 40 token/s generation)


| Series                | Cores         | L3 Cache   | PCIe Lanes | Max RAM      |
|-----------------------|---------------|------------|------------|--------------|
| Threadripper 7000X    | 24-64         | 128 MB     | 48         | 512 GB       |
| Threadripper Pro 7000WX | 12-96       | 384 MB     | 128        | 2 TB         |
| EPYC 8004             | 8-64          | 128 MB     | 96         | 768 GB       |
| Threadripper 9000X    | 24-64         | 128 MB     | 48         | 512 GB       |
| Threadripper Pro 9000WX | 12-96       | 384 MB     | 128        | 2 TB         |

### Server CPUs

| Series                | Cores         | L3 Cache   | Arch    | Socket      | PCIe Lanes | Max RAM      |
|-----------------------|---------------|------------|---------|-------------|------------|--------------|
| EPYC 9004             | 16-96         | 384 MB     | Zen 4   | SP5         | 128        | 6 TB         |
| EPYC 9005             | 16-128        | 384 MB     | Zen 5   | SP5         | 128        | 6 TB         |

###  Mobile CPUs

| Series                | Cores         | L3 Cache   | Arch    | Socket      | PCIe Lanes | Max RAM      |
|-----------------------|---------------|------------|---------|-------------|------------|--------------|
| Ryzen 7040/7045/8040/8045/200 | 4-8   | 16 MB      | Zen 4   | FP7/FP7r2/FP8/FL1 | 20   | 64 GB        |
| Ryzen 200             | 4-8           | 16 MB      | Zen 4   | FP7/FP7r2/FP8 | 20       | 64 GB        |
| Ryzen AI MAX/MAX+ 300 | 4-12          | 24 MB      | Zen 5c  | FP8         | 20         | 64 GB        |


**Notes:**
- See


Notes for CPU-only configurations:



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

|                        | **Embedded CPU + NPU** |
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



Notes:
