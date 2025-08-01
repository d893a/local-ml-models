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

General notes:
-   CPU RAM: In all cases the CPU RAM needs to be large enough to hold the model and context.


### CPU only

Notes for CPU-only configurations:
-   Only AMD CPUs are considered in this assessment.
-   CPU Architecture: bfloat16 format is needed for LLM inference. This was introduced with the
    AMD Zen 4 architecture's AVX-512 instruction set.
    -   [Zen 4/4c](https://en.wikipedia.org/wiki/Zen_4) CPUs:
        -   Desktop:
            -   Ryzen 7700/7900/7950/7040/7045/8000F/8000G (AM5 socket)
        -   Workstation:
            -   Ryzen Threadripper 7000X / Threadripper Pro 7000WX (sTR5 socket)
        -   Server:
            -   EPYC 4004 (AM5)
            -   EPYC 8004 (SP6)
            -   EPYC 9004 (SP5)
        -   Mobile:
            -   Ryzen 7040/7045/8040/8045/200 (FP7/FP7r2/FP8/FL1)
            -   Ryzen 200 (BGA FP7/FP7r2/FP8)
    -   [Zen 5/5c](https://en.wikipedia.org/wiki/Zen_5) CPUs: , Threadripper (Pro) 9000, Ryzen AI (MAX,MAX+) 3xx, EPYC 9005
        -   Desktop:
            -   Ryzen 9000 (AM5 socket)
        -   Workstation:
            -   Ryzen Threadripper 9000X / Threadripper Pro 9000WX (sTR5 socket)
        -   Server:
            -   EPYC 4004 (AM5)
            -   EPYC 8004 (SP6)
            -   EPYC 9004 (SP5)
        -   Mobile:
            -   Ryzen AI 300 (BGA FP8)
-   CPU core count:
-   CPU socket:
    -   Consumer CPUs use the AM5 socket. Wide motherboard support.
    -   Workstation CPU sockets:
        -   sTR5: Threadripper, Threadripper Pro
        -   SP6: AMD EPYC 8004 series
    -   Server CPUs: SP5 socket
-   CPU Cache size:
    -   Zen 4/4c CPUs:
        -   Ryzen 7000/8000, Hawk Point Refresh (2xx), Threadripper (Pro) 7000, EPYC 4004/8004/9004
        -   L1 cache: 32+32 kB, L2: 1MB, L3: 32 MB
    -   Zen 5/5c CPUs:
        -   Ryzen 9000, Threadripper (Pro) 9000, Ryzen AI (MAX,MAX+) 3xx, EPYC 9005
        -   L1 cache: 32+48 kB, L2: 1MB, L3: 32 MB

1.  FCLK: Fabric clock speed - clock speed on the CPU side. AMD Zen 4: 1.8 GHz; Zen 5: 2.0 GHz.
2.  CCD: Core complex Die - contains the memory controller and caches in the CPU.
3.  RAM Bandwidth B = min(RAM channels * 8 * RAM speed [MT/s] / 1000, Number of CCDs * 32 * FCLK [GHz]) where FCLK is the fabric clock speed.


|                        | **Consumer<br>CPU only**      | **Workstation<br>CPU only**      | **Server<br>CPU only**      |
|------------------------|------------------------------|----------------------------------|-----------------------------|
| **CPU [AMD]**          | Ryzen                        | Threadripper /<br>Threadripper Pro | EPYC                        |
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



### CPU + GPU

|                        | **Consumer CPU+<br>Consumer GPU**      | **Workstation<br>GPU**      | **Server<br>GPU**      |
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
