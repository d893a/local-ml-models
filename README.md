## Introduction

This is a collection of links and information to assess the viability of running machine learning models and large language models locally.

The aim is to support engineers and stakeholders to make a well-informed decision when procuring LLM infrastructure.

Running machine learning models requires:
- Hardware
- Models
- Software


## Hardware

Different use cases require different hardware. The following hardware categories are considered.

LLM inference:
-   CPU only
-   CPU + GPU (The model plus context fits into the GPU memory)
-   Hybrid CPU + GPU, including embedded (Mini PC) solutions
    The model plus context fits into the CPU memory, but does not fit into the GPU VRAM


### CPU only

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

---

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
1.  FCLK: Fabric clock speed - clock speed on the CPU side. AMD Zen 4: 1.8 GHz; Zen 5: 2.0 GHz.
1.  The bfloat16 format for LLM inference is supported by the AMD CPUs in the
    AVX-512 instruction set starting with the Zen 4 architecture.
    -   Zen 4/4c CPUs:
        -   Ryzen 7000/8000, Hawk Point Refresh (2xx), Threadripper (Pro) 7000, EPYC 4004/8004/9004
        -   L1 cache: 32+32 kB, L2: 1MB, L3: 32 MB
    -   Zen 5/5c CPUs:
        -   Ryzen 9000, Threadripper (Pro) 9000, Ryzen AI (MAX,MAX+) 3xx, EPYC 9005
        -   L1 cache: 32+48 kB, L2: 1MB, L3: 32 MB
2.  CCD: Core complex Die - contains the memory controller and caches in the CPU.
4.  RAM Bandwidth B = min(RAM channels * 8 * RAM speed [MT/s] / 1000, Number of CCDs * 32 * FCLK [GHz]) where FCLK is the fabric clock speed.