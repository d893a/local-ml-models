# Hardware alternatives for running machine learning models locally

-   [Server](#server)
-   [Light server](#light-server)
-   [Workstation](#workstation)
-   [Desktop PC](#desktop-pc)
-   [Embedded / mobile CPU](#embedded--mobile-cpu)
-   [GPUs](#gpus)

Refer to the *[Which GPU(s) to Get for Deep Learning: My Experience
and Advice for Using GPUs in Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)*
blog post by Tim Dettmers.
>   Tensor Cores are most important, followed by memory bandwidth of a GPU,
>   the cache hierarchy, and only then FLOPS of a GPU

See also the [Build your own machine](https://huggingface.co/docs/transformers/perf_hardware) guide on HuggingFace.

## Server

Minimal 1-CPU configuration
-   Supports 2x NVIDIA RTX PRO 6000 Blackwell (96GB) Desktop GPUs with open-air cooling
-   Theoretical maximum RAM bandwidth of 460.8 GB/s
-   Suggested processors: AMD EPYC 9354, 9534.

| Component   | Model                                              | Price each<br>[HUF] | Price subtotal<br>[HUF] |
|-------------|----------------------------------------------------|--------------------:|------------------------:|
| CPU         | AMD EPYC 9354                                      |       1,120,491     |           1,120,491     |
| RAM         | Micron 32GB DDR5 5600MHz MTC20F2085S1RC56BR × 12   |          76,020     |             912,240     |
| SSD         | Samsung PM9A3 1.9TB NVMe PCIe Gen4 V6 M.2 22x110   |         129,910     |             129,910     |
| Motherboard | Supermicro MBD-H13SSL-NT                           |         328,912     |             328,912     |
| CPU cooler  | Arctic Freezer 4U-SP5                              |          23,990     |              23,990     |
| PSU         | Seasonic Prime PX-2200 2200W 80 PLUS Platinum      |         212,990     |             212,990     |
| Chassis     | Fractal Design Torrent                             |          75,600     |              75,600     |
| **Total**   |                                                    |                     |         **2,804,133**   |

1-CPU maximum configuration with Zen 4 architecture
-   Supports 2x NVIDIA RTX PRO 6000 Blackwell (96GB) Desktop GPUs with open-air cooling,
-   Theoretical maximum RAM bandwidth of 460.8 GB/s
-   Suggested processors:
    -   9754 (128 cores, L3 cache 256 MB),
    -   9184X, 9384X, 9684X (16/32/96 cores, L3 cache 768/768/1152 MB)

| Component   | Model                                              | Price each<br>[HUF] | Price subtotal<br>[HUF] |
|-------------|----------------------------------------------------|--------------------:|------------------------:|
| CPU         | *AMD EPYC 9754*                                    |      *2,891,190*    |          *2,891,190*    |
| RAM         | Micron 32GB DDR5 5600MHz MTC20F2085S1RC56BR × 12   |          76,020     |             912,240     |
| SSD         | Samsung PM9A3 1.9TB NVMe PCIe Gen4 V6 M.2 22x110   |         129,910     |             129,910     |
| Motherboard | Supermicro MBD-H13SSL-NT                           |         328,912     |             328,912     |
| CPU cooler  | Arctic Freezer 4U-SP5                              |          23,990     |              23,990     |
| PSU         | Seasonic Prime PX-2200 2200W 80 PLUS Platinum      |         212,990     |             212,990     |
| Chassis     | Fractal Design Torrent                             |          75,600     |              75,600     |
| **Total**   |                                                    |                     |         **4,574,832**   |

<!--
2-CPU configuration optimized for Zen 4 CPU inference
-   Supports 2x NVIDIA RTX PRO 6000 Blackwell (96GB) Desktop GPUs with open-air cooling,
-   Theoretical maximum RAM bandwidth of 460.8 GB/s

| Component   | Model                                              | Price each<br>[HUF] | Price subtotal<br>[HUF] |
|-------------|----------------------------------------------------|--------------------:|------------------------:|
| CPU         | *AMD EPYC 9754 x 2*                                |      *1,243,230*    |          *2,486,460*    |
| RAM         | min DDR5 4800 RDIMM x 24                           |                     |                         |
| SSD         | Samsung PM9A3 1.9TB NVMe PCIe Gen4 V6 M.2 22x110   |         129,910     |             129,910     |
| Motherboard |                                                    |         328,912     |             328,912     |
| CPU cooler  | Arctic Freezer 4U-SP5?                             |          23,990     |              23,990     |
| PSU         |                                                    |                     |                         |
| Chassis     |                                                    |                     |                         |
| **Total**   |                                                    |                     |         **2,926,872**   |
-->

<!--
1-CPU configuration optimized for Zen 5 CPU inference
-   Support 2x NVIDIA RTX PRO 6000 Blackwell (96GB) Desktop GPUs with open-air cooling,
-   Theoretical maximum RAM bandwidth of 576.0 GB/s

| Component   | Model                                              | Price each<br>[HUF] | Price subtotal<br>[HUF] |
|-------------|----------------------------------------------------|--------------------:|------------------------:|
| CPU         | AMD EPYC 9175F                                     |       1,130,275     |           1,130,275     |
| RAM         | Micron 32GB 2Rx8 6400MHz MTC20F2085S1RC64BR        |          89,093     |           1,069,116     |
| SSD         | Samsung PM9A3 1.9TB NVMe PCIe Gen4 V6 M.2 22x110   |         129,910     |             129,910     |
| Motherboard | Supermicro MBD-H13SSL-NT                           |         328,912     |             328,912     |
| CPU cooler  | Arctic Freezer 4U-SP5                              |          23,990     |              23,990     |
| PSU         | Seasonic Prime PX-2200 2200W 80 PLUS Platinum      |         212,990     |             212,990     |
| Chassis     | Fractal Design Torrent                             |          75,600     |              75,600     |
| **Total**   |                                                    |                     |         **2,970,793**   |
-->

### Specs

<details><summary>CPU: AMD EPYC 9004 / 9005 (SP5 socket)</summary>

-   Minimum 12 x DDR5 4800 MT/s RAM
    -   RAM module side memory bandwidth:
        -   [1-CPU config](https://hothardware.com/Image/Resize/?width=1170&height=1170&imageFile=/contentimages/Article/3257/content/big_epyc-cpu-memory-capabilities.jpg):
            12 x 8 x 4.8 GT/s = 460.8 GB/s
            -   Supports 6x PCIe 5.0 x16 GPUs
        -   2-CPU config: 24 x 8 x 4.8 GT/s = 921.6 GB/s
-   Minimum 8 CCDs per processor:
    -   CPU side memory bandwidth:
        -   Zen 4: 8 x 32 x 1.8 GHz = 460.8 GB/s
        -   Zen 5: 8 x 32 x 2.0 GHz = 512.0 GB/s
        -   Supports 6x PCIe 5.0 x16 GPUs (6 x 63 = 378 GB/s)
-   CPU candidates which support PCIe 5.0 x16:
    -   [AMD EPYC 9004 series processors](https://www.amd.com/content/dam/amd/en/documents/products/epyc/epyc-9004-series-processors-data-sheet.pdf) \
        ![EPYC 9004 Series CPU Positioning](<assets/AMD EPYC 9004 series processors.png>) \
        [Zen 4](https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)):
        Any AMD EPYC 9004 CPU, except 9124, 9224, 9254, 9334, because n_CCD < 8. Minimum viable: 9354 / 9354P. Performance and TDP tuned: 9754
    -   [AMD EPYC 9005 series processors](https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/datasheets/amd-epyc-9005-series-processor-datasheet.pdf) \
        ![AMD EPYC 9005 series processors](<assets/AMD EPYC 9005 series processors.png>) \
        [Zen 5](https://en.wikipedia.org/wiki/Epyc#Fifth_generation_Epyc_(Grado,_Turin_and_Turin_Dense)):
        Any AMD EPYC 9005 CPU, except 9015, 9115, 9135, 9255, 9335, 9365, because n_CCD < 8. Minimum viable: 9355 / 9355P. Performance and TDP tuned: 9745
    -   Complete list of CPU candidates:
        -   Zen 4: 9174F, 9184X, 9274F, 9354, 9354P, 9374F, 9384X,
            9454, 9454P, 9474F, 9534, 9554, 9554P, 9634, 9654, 9654P,
            9684X, 9734, 9754, 9754S
        -   Zen 5: 4245P, 4345P, 4465P, 4545P, 4565P, 4585PX, 9175F,
            9275F, 9355, 9355P, 9375F, 9455, 9455P, 9475F, 9535, 9555,
            9555P, 9565, 9575F, 9645, 9655, 9655P, 9745, 9755, 9825,
            9845, 9965
    -   If CPU inference is not a priority, then lower core count and
        thus lower DTP/cDTP is sufficient.
    -   All CPUs below the 240 TDP line have less than 8 CCDs, so they cannot utilize the available RAM bandwidth.
    -   CPU candidates whose configurable TDP (cTDP) is
        [in the 240-300 W range](https://www.amd.com/en/products/specifications/server-processor.html):
        -   Zen 4: 9354, 9354P, 9454, 9454P, 9534, 9634
        -   Zen 5: 9355P, 9355, 9365, 9455P, 9455, 9535
    -   CPU candidates whose configurable TDP (cTDP) is
        [above 300 W](https://www.amd.com/en/products/specifications/server-processor.html):
        -   Zen 4: 9174F, 9184X, 9274F, 9374F, 9384X, 9474F, 9554,
            9554P, 9654, 9654P, 9684X, 9734, 9754, 9754S
        -   Zen 5: 9175F, 9275F, 9375F, 9475F, 9555, 9555P, 9565,
            9575F, 9645, 9655, 9655P, 9745, 9825, 9845
-   **1-CPU minimal pick: AMD EPYC 9354**
    ([Wikipedia](https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)))
    ([AMD](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9354.html))
    ([TechPowerUp](https://www.techpowerup.com/cpu-specs/epyc-9354.c2923))
    -   Cores: 32
    -   TDP: 280 W
    -   Configurable TDP: 240-300 W
    -   Max. Memory: 12 x 128 = 1536 GB
    -   PCI-Express: Gen 5, 128 Lanes (CPU only)
    -   Rated Memory Speed: 4800 MT/s
    -   Max memory bandwidth:
        -   12 channels x 8 x 4.8 = 460.8 GB/s,
        -   8 CCD x 32 x 1.8 GHz FCLK = 460.8 GB/s
    -   Cache L3: 256 MB (shared)
-   **Ideal pick for performance and low TDP: AMD EPYC 9754**
    ([Wikipedia](https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)))
    ([AMD](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9754.html))
    ([TechPowerUp](https://www.techpowerup.com/cpu-specs/epyc-9754.c3257))
    -   Cores: 128
    -   TDP: 360 W
    -   Configurable TDP: 320-400 W
    -   Max. Memory: 12 x 256 = 3072 GB
    -   PCI-Express: Gen 5, 128 Lanes
    -   Rated Memory Speed: 4800 MT/s
    -   1-CPU configuration maximum memory bandwidth:
        -   12 channels x 8 x 4.8 = 460.8 GB/s,
        -   8 CCD x 32 x 1.8 GHz FCLK = 460.8 GB/s
        -   Theoretical maximum bandwidth: 460.8 GB/s
    -   2-CPU configuration maximum memory bandwidth:
        -   24 channels x 8 x 4.8 GT/s = 921.6 GB/s
        -   2 x 8 CCD x 32 x 2.0 GHz FCLK = 1024 GB/s
        -   Theoretical maximum bandwidth: 921.6 GB/s
        -   A 2-CPU configuration would require a server chassis with multiple
            PSUs to deliver 2x 400 W (CPUs) + 2x 600 W (GPUs) + headroom = 3000 W total.
    -   Cache L3: 256 MB (shared)
    <!--
    ([AMD](https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9754.html))
    -   Rated Memory Speed: 6000 MT/s ([6400 MT/s for certain validated systems](https://chipsandcheese.com/p/amds-turin-5th-gen-epyc-launched))
    -   1-CPU configuration maximum memory bandwidth:
        -   12 channels x 8 x 6 GT/s = 576 GB/s
        -   16 CCD x 32 x 2.0 GHz FCLK = 1024 GB/s
        -   Theoretical maximum bandwidth: 576 GB/s
    -   2-CPU configuration maximum memory bandwidth:
        -   24 channels x 8 x 4.4 GT/s = 844.8 GB/s
        -   2 x 16 CCD x 32 x 2.0 GHz FCLK = 2048 GB/s
        -   Theoretical maximum bandwidth: 844.8 GB/s
    -   Cache L3: 512 MB (shared)
    -   Each CCX in the 9175F likely 32 MB of L3 cache (512 MB total ÷ 16
        CCXs). This means each core has its own private 32 MB of L3 -- no
        competition with other cores for cache space. Result: excellent data
        locality and low cache contention in memory-intensive single-threaded
        or lightly-threaded workloads.
    -->

</details>
<details><summary>CPU cooler</summary>

- [Arctic Freezer 4U-SP5](https://www.arctic.de/en/Freezer-4U-SP5/ACFRE00158A)
    - Thermal compound: ARCTIC MX-6 0.8 g syringe included
    - Operating ambient temperature: 0–40 °C
    - Dimensions: 124 mm (L) × 147 mm (W) × 145 mm (H)
    - Weight: 1512 g
    - Compatibility: AMD SP5, server rack unit 4U and up
    - Heatsink:
        - Direct touch heat pipes, 10 × Ø 6 mm
        - 62 aluminum fins
    - Fans:
        - 2 × 120 mm PWM fans
        - Speed: 300–3300 rpm (PWM controlled)
        - Connector: 4-pin plug, 200 mm cable
        - Bearing: dual ball bearing
        - Noise level: 45.3 dBA
        - Air flow: 81.04 cfm (137.69 m³/h)
        - Static pressure: 4.35 mmH₂O
        - Current/voltage: 0.29 A / 12 V DC
        - Startup voltage: 3.1 V DC
-   [Silverstone XE360-SP5](https://www.silverstonetek.com/en/product/info/coolers/xe360_sp5/)
    -   High Performance Triple 120mm All-In-One Liquid Cooler for AMD Socket SP5
-   [Silverstone XED120 WS](https://www.silverstonetek.com/en/product/info/coolers/xed120s_ws/)
    -   4U Form Factor Industrial-Grade CPU Cooler with TDP 450W for Intel & AMD Server-Grade Sockets
    -   Model No.: SST-XED120S-WS
    -   Material: Copper heat pipes with aluminum fins
    - Application:
        - Intel LGA4677/4710 (CPU carrier not included)
        - AMD Socket SP5, SP6, sTR5, SP3, TR4, sWRX8, sWRX9
    - Fan
    - Dimensions:
        - 120mm (W) x 30mm (H) x 120mm (D)
        - 4.72" (W) x 1.18" (H) x 4.72" (D)
    - Speed: 500–3000 RPM
    - Noise level: **44.9 dBA**
    - Rated voltage: 12V
    - Rated current: 0.35A
    - Maximum airflow: 102 CFM
    - Maximum air pressure: 8.24 mmH2O
    - Connector: 4-pin PWM
    - Bearing: Dual ball bearing
    - MTTF: 70,000 hours
    - CPU TDP support: up to 450W
    - Dimensions (with cooler): 120mm (W) x 145mm (H) x 120mm (D)
        - 4.72" (W) x 5.71" (H) x 4.72" (D)
-   [Silverstone XE04-SP5](https://www.silverstonetek.com/en/product/info/coolers/xe04_sp5/)
    - 4U form factor server/workstation small form factor CPU cooler for AMD SP5 sockets
    - Model numbers:
        - SST-XE04-SP5 (Silver+Black)
        - SST-XE04-SP5B (Black+Black)
    - Material: aluminum fins and heat pipes
    - Application: AMD Socket SP5
    - Fan dimensions: 92mm (W) x 25mm (H) x 92mm (D)
    - Fan speed: 1500 to 5000 RPM
    - Noise level: **43 dBA at full speed**
    - Rated voltage: 12V
    - Rated current: 0.66A
    - Maximum airflow: 77.7 CFM
    - Maximum air pressure: 10.67 mmH2O
    - Connector: 4-pin PWM
    - Bearing type: dual ball bearing
    - MTTF: 90,000 hours
    - CPU TDP support: up to 400W
    - Cooler dimensions: 93mm (W) x 128mm (H) x 118mm (D)

</details>
<details><summary>RAM</summary>

-   The target system must support at least 2 NVIDIA RTX PRO 6000 Blackwell (96GB) GPUs
    -   Required system RAM: >> total GPU VRAM
        -   2x 96 GB GPUs: 192 GB minimum, 384 GB ideally
        -   4x 96 GB GPUs: 384 GB minimum, 768 GB ideally
        -   6x 96 GB GPUs: 576 GB minimum, 1152 GB ideally
    -   Memory modules:
        -   Note: From the Genoa (AMD EPYC 4004, 8004, 9004) platform on,
            [single-rank memory modules will perform well](https://semianalysis.com/2022/11/10/amd-genoa-detailed-architecture-makes/)
            >   The other important feature is dual rank versus single rank memory.
            >   With Milan and most Intel platforms, dual-rank memory is crucial to
            >   maximizing performance. There’s a 25% performance delta on Milan,
            >   for example. With Genoa, this is brought down to 4.5%. This is
            >   another considerable cost improvement because cheaper single-rank
            >   memory can be used.
        ([Slide](https://i0.wp.com/semianalysis.com/wp-content/uploads/2024/11/https3A2F2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com2Fpublic2Fimages2F8aba2a1b-dc51-41c5-a618-3ad93dfcd169_5278x2891-scaled.jpg?ssl=1))
    -   Required system RAM bandwidth: min 63 GB/s per GPU (due to PCIe x16 bus bandwidth)
        -   2 GPUs: min. 126 GB/s
        -   4 GPUs: min. 252 GB/s
        -   6 GPUs: min. 378 GB/s
        -   Lower RAM bandwidth will work, but not at full performance
-   [AMD EPYC 9004 Series Memory Population Recommendations](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/user-guides/amd-epyc-9004-ug-memory-population-recommendations.pdf)
    -   4th Gen AMD EPYC processors support memory with the following characteristics:
        -   RDIMM: 16GB 1Rx8, 24GB 1Rx8, 32GB 1Rx4, 32GB 2Rx8, 40GB 2Rx8, 48GB 1Rx4, 48GB 2Rx8, 64GB 2Rx4, 80GB 2Rx4, 96GB 2Rx4
        -   3DS RDIMM: 128GB 2S2Rx4, 192GB 2S2Rx4, 256GB 2S4Rx4, 384GB 2S4Rx4, 512GB 2S8R (pending ecosystem enablement)
        -   ECC: 80b x4, 80b x8, 72b x4.
        -   Optimized Bounded Fault ECC DRAM: 80b x4 AMDC, 80b x8, 72b x4
        -   Use the same memory configuration for all NUMA domains in a single processor socket when using NPS=2 or NPS=4. “NPS” = NUMA node(s) per socket.
        -   [Table 2-1 shows recommended memory speeds for a variety of memory types](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/user-guides/amd-epyc-9004-ug-memory-population-recommendations.pdf)
            ![alt text](<assets/AMD EPYC 9004 Processor Memory Population Recommendations.Recommended memory speeds.png>)
            >   RDIMMs built from x4 and x8 devices are supported. The
            >   frequencies shown apply to both. The capacities listed only
            >   represent those of x4 DIMMs. RDIMMs built with x8 devices have
            >   half the capacity of the x4 RDIMMs with an equal number of
            >   ranks.
            >
            >   The following DIMM types are not supported: LRDIMM, UDIMM, NVDIMM-N, NVDIMM-P
            >
            >   All DIMM modules must be RDIMM or RDIMM 3DS module types with
            >   the same ECC configuration. Do not mix DIMM module types
            >   within a memory channel. Do not mix x4 and x8 DIMMs within a
            >   memory channel. Do not mix 3DS and non-3DS memory modules in a
            >   2DPC system.
-   Tested memory list for the [Supermicro H13SSL‑NT](https://www.supermicro.com/en/products/motherboard/H13SSL-NT) motherboard (see under "Resources"):
    | Part Number         | Description                              | Compatible Motherboard Revision(s) |
    |---------------------|------------------------------------------|------------------------------------|
    | MEM-DR532MD-ER64    | 32GB DDR5-6400 2Rx8 (16Gb) ECC RDIMM     | R2.01 and above                    |

    -   NOTE: 6400 MT/s speed requires motherboard revision 2.01 or higher and
        BIOS v3.4 or newer. See FAQ
        [#43110](https://www.supermicro.com/support/faqs/faq.cfm?faq=43110)
        for more information.

-   [Samsung candidates](https://semiconductor.samsung.com/dram/module/rdimm/#finder): RDIMM, DDR5, 32GB
    | Part Number      | Speed      | Org      | Density             |
    |------------------|------------|----------|---------------------|
    | M321R4GA0BB0-CQK | 4800 Mbps  | 1R x 4   | (4G x 4) x 20       |
    | M321R4GA3BB6-CQK | 4800 Mbps  | 2R x 8   | (2G x 8) x 20       |
    | M329R4GA0BB0-CQK | 4800 Mbps  | 1R x 4   | (4G x 4) x 18 (9x4) |
    | M321R4GA0EB2-CWM | 5600 Mbps  | 1R x 4   | (4G x 4) x 20       |
    | M321R4GA3EB2-CWM | 5600 Mbps  | 2R x 8   | (2G x 8) x 20       |
    | M321R4GA0EB0-CWM | 5600 Mbps  | 1R x 4   | (4G x 4) x 20       |
    | M321R4GA3EB0-CWM | 5600 Mbps  | 2R x 8   | (2G x 8) x 20       |
    | M321R4GA0PB0-CWM | 5600 Mbps  | 1R x 4   | (4G x 4) x 20       |
    | M321R4GA3PB0-CWM | 5600 Mbps  | 2R x 8   | (2G x 8) x 20       |
    | M321R4GA0EB2-CCP | 6400 Mbps  | 1R x 4   | (4G x 4) x 20       |
    | M321R4GA3EB2-CCP | 6400 Mbps  | 2R x 8   | (2G x 8) x 20       |
    | M321R4GA0PB2-CCP | 6400 Mbps  | 1R x 4   | (4G x 4) x 20       |
    | M321R4GA3PB2-CCP | 6400 Mbps  | 2R x 8   | (2G x 8) x 20       |
-   [Kingston candidates](https://www.kingston.com/en/memory/server-premier?memory=ddr5&dimmtype=registered&capacity=32)
    | Part Number           | Capacity | Org | Ranking | DRAM MFR | Speed MT/s |
    |-----------------------|----------|-----|---------|----------|-----------:|
    | KSM48R40BD8-32HA      | 32GB     | X8  | 2R      | Hynix    | 4800       |
    | KSM48R40BD8-32MD      | 32GB     | X8  | 2R      | Micron   | 4800       |
    | KSM56R46BD8-32HA      | 32GB     | X8  | 2R      | Hynix    | 5600       |
    | KSM56R46BD8-32MD      | 32GB     | X8  | 2R      | Micron   | 5600       |
    | KSM56R46BD8PMI-32HAI  | 32GB     | X8  | 2R      | Hynix    | 5600       |
    | KSM56R46BD8PMI-32MDI  | 32GB     | X8  | 2R      | Micron   | 5600       |
    | KSM56R46BS4PMI-32HAI  | 32GB     | X4  | 1R      | Hynix    | 5600       |
    | KSM56R46BS4PMI-32MDI  | 32GB     | X4  | 1R      | Micron   | 5600       |
    | KSM64R52BD8-32MD 	    | 32GB     | X8  | 2R      | 16Gbit   | 6400       |
-   [Micron candidates](https://www.crucial.com/catalog/memory/server?selectedValues=DDR5-4800@speed--DDR5-5600@speed--DDR5-6400@speed--RDIMM@module_type--DDR5@technology--32GB@density)
    | Part Number        | Model                       | Capacity | Speed | Type   | Rank | CL  |
    |--------------------|-----------------------------|----------|-------|--------|------|-----|
    | MTC20F2085S1RC48BR | Micron DDR5-4800 RDIMM 2Rx8 | 32 GB    | 4800  | RDIMM  | 2Rx8 | 40  |
    | MTC20F1045S1RC48BR | Micron DDR5-4800 RDIMM 1Rx4 | 32 GB    | 4800  | RDIMM  | 1Rx4 | 40  |
    | MTC20F2085S1RC56BR | Micron DDR5-5600 RDIMM 2Rx8 | 32 GB    | 5600  | RDIMM  | 2Rx8 | 46  |
    | MTC20F1045S1RC56BR | Micron DDR5-5600 RDIMM 1Rx4 | 32 GB    | 5600  | RDIMM  | 1Rx4 | 46  |

</details>
<details><summary>Motherboard</summary>

-   [Supermicro H13SSL‑NT](https://www.supermicro.com/en/products/motherboard/H13SSL-NT) /
    [Supermicro H13SSL‑N](https://www.supermicro.com/en/products/motherboard/h13ssl-n):
    -   Form Factor: ATX
    -   LAN: N: 2x 1 Gbps LAN; NT: 2x 10 Gbps
    -   12 DDR5 slots,
    -   up to 3TB RAM support, and
    -   robust PCIe layout for multi‑GPU.
    -   CPU: AMD EPYC™ 9004 series Processors
        -   Single Socket SP5 supported, CPU TDP supports Up to 400W TDP
    -   System Memory
        -   Memory Capacity: 12 DIMM slots
        -   Up to 3TB 3DS ECC Registered RDIMM, DDR5-4800MHz
    -   Memory Type: 4800 MT/s ECC DDR5 RDIMM (3DS)
        -   Up to 256GB of memory with speeds of up to 4800MHz (1DPC)
    -   DIMM Sizes: 16GB, 24GB, 32GB, 40GB, 48GB, 64GB, 80GB, 96GB, 128GB, 192GB, 256GB
    -   Memory Voltage: 1.1V
    -   Network Controllers: Dual LAN with Broadcom BCM57416 10GBase-T
    -   Input / Output
        -   SATA: 8 SATA3 (6Gbps) port(s)
        -   LAN: 1 RJ45 Dedicated IPMI LAN port
        -   USB: 6 USB 3.0 port(s) (4 USB; 2 via header)
        -   Video Output: 1 VGA port(s)
        -   Serial Port: 1 COM Port(s) (1 header)
        -   TPM: 1 TPM Header
        -   Others
            -   1 MCIO (PCIe 5.0 x8/SATA3) Port(s)
            -   2 MCIO (PCIe 5.0 x8) Port(s)
    -   Expansion Slots
        -   PCIe
            -   3 PCIe 5.0 x16 (in x16 slot),
            -   2 PCIe 5.0 x8
        -   M.2
            -   M.2 Interface: 2 SATA/PCIe 4.0 x4
            -   Form Factor: 2280/22110
            -   Key: M-Key
    -   Widely used in community for stable performance ([Newegg.com][3.12]).
-   [ASRock Rack GENOAD8QM3‑2T/BCM](https://www.asrockrack.com/general/productdetail.asp?Model=GENOAD8QM3-2T/BCM#Specifications):
    -   Not suitable: **Only 8 DIMM slots (1DPC)**
-   [ASRock Rack GENOAD8UD‑2T/X550](https://www.asrockrack.com/general/productdetail.asp?Model=GENOAD8UD-2T/X550#Specifications):
    -   Not suitable: **Only 8 DIMM slots (1DPC)**
-   [ASRock Rack GENOAD24QM3-2L2T/BCM](https://www.asrockrack.com/general/productdetail.asp?Model=GENOAD24QM3-2L2T%2FBCM&utm_source=GENOA+launch&utm_medium=11_landing+page#Specifications)
    -   EEB (12" x 14")
    -   Single Socket SP5 (LGA 6096), supports AMD EPYC™ 9005*/9004 (with AMD 3D V-Cache™ Technology) and 97x4 series processors
    -   24 DIMM slots (2DPC), supports DDR5 RDIMM, RDIMM-3DS
    -   2 PCIe5.0 x16
    -   7 MCIO (PCIe5.0 x8), 2 MCIO (PCIe5.0 x8 or 8 SATA 6Gb/s)
    -   Supports 2 M.2 (PCIe5.0 x4)
-   [GIGABYTE MZ33-AR0](https://www.gigabyte.com/Enterprise/Server-Motherboard/MZ33-AR0-rev-1x-3x)
    -   Form Factor: E-ATX, 305 x 330
    -   CPU
        -   AMD EPYC™ 9005 Series Processors
        -   AMD EPYC™ 9004 Series Processors
        -   Single processor, cTDP up to 400W
    -   Socket: 1 x LGA 6096 Socket SP5
    -   Memory
        -   24 x DIMM slots, DDR5 memory supported
        -   12-Channel memory per processor
        -   AMD EPYC™ 9005:
            -   RDIMM: Up to 4800 MT/s (1DPC)
            -   RDIMM: Up to 4000 MT/s (1R 2DPC), 3600 MT/s (2R 2DPC)
        -   AMD EPYC™ 9004:
            -   RDIMM: Up to 4800 MT/s (1DPC), 3600 MT/s (2DPC)
    -   LAN: 2 x 10Gb/s LAN (1 x Broadcom® BCM57416)
        -   Support NCSI function
        -   1 x 10/100/1000 Mbps Management LAN
    -   Storage Interface
        -   MCIO:
            -   2 x MCIO 8i for 4 x Gen5 NVMe or 16 x SATA
            -   4 x MCIO 8i for 8 x Gen5 NVMe
            -   1 x MCIO 8i for 2 x Gen4 NVMe
        -   M.2:
            -   1 x M.2 (2280/22110), PCIe Gen4 x4
        -   RAID: N/A
-   [GIGABYTE MZ33-CP1](https://www.gigabyte.com/Enterprise/Server-Motherboard/MZ33-CP1-rev-3x)
    -   Single AMD EPYC™ 9005/9004 Series Processors
    -   12-Channel DDR5 RDIMM, 12 x DIMMs
    -   2 x 1Gb/s LAN ports via Intel® I210-AT
    -   4 x MCIO 8i connectors with PCIe Gen5 x8 interface
    -   2 x MCIO 8i connectors with PCIe Gen4 x8 or SATA interface
    -   1 x M.2 slot with PCIe Gen3 x4 interface
    -   3 x PCIe Gen5 x16 expansion slots
    -   1 x PCIe Gen4 x16 expansion slot
    -   1 x OCP NIC 3.0 PCIe Gen5 x16 slot
    -   Memory
        -   12 x DIMM slots
        -   DDR5 memory supported
        -   12-Channel memory per processor
        -   AMD EPYC™ 9005: RDIMM: Up to 6400 MT/s
        -   AMD EPYC™ 9004: RDIMM: Up to 4800 MT/s
-   [GIGABYTE MZ73-LM2](https://www.gigabyte.com/us/Enterprise/Server-Motherboard/MZ73-LM2-rev-3x)
    -   Dual AMD EPYC™ 9005/9004 Series Processors
    -   12-Channel DDR5 RDIMM, 24 x DIMMs
    -   2 x 10Gb/s LAN ports via Broadcom® BCM57416
    -   2 x MCIO 8i connectors with PCIe Gen5 x8 or SATA interface
    -   1 x SlimSAS 4i connector with SATA interface
    -   1 x M.2 slot with PCIe Gen5 x4 interface
    -   4 x PCIe Gen5 x16 expansion slots
    -   E‑ATX form factor, includes
    -   multiple PCIe 5 slots (x16) spaced for GPUs.
    -   Reddit warns of interference issues between memory and GPU slots on some layouts - ASRock GENOA variants often preferred ([Reddit][3.13]).
    -   [ServeTheHome forum post](https://forums.servethehome.com/index.php?threads/motherboard-for-dual-epyc-9965-one-that-actually-works-and-fits-a-5090-gpu.48129/#post-471063):
        >   GigaByte MZ73-LM2 Rev. 3.x E-ATX: as well as its predecessors
        >   have the famous ""WAIT FOR CHIPSET TO INITIALIZE"" issue and
        >   the Gigabyte support that let's you down.
-   [Asus K14PA-U12](https://servers.asus.com/products/servers/server-motherboards/K14PA-U12#Specifications)
    -   Form Factor: CEB, 12" x 10.5"
    -   Processor / System Bus 1 x Socket SP5 (LGA 6096)
    -   AMD EPYC™ Genoa Processor (up to 400W)
    -   Memory
        -   Total Slots: 12 (12-channel, 1-DIMM per Channel)
        -   Capacity:Maximum up to 3TB
        -   Memory Type: DDR5 4800 RDIMM/3DS RDIMM
        -   Memory Size: 256GB, 128GB, 96GB, 64GB, 48GB, 32GB, 24GB, 16GB (RDIMM) (RDIMM)
            -   Please refer to www.asus.com for latest memory AVL update
    -   Expansion Slots:
        -   Total Slot : 3
        -   3 x PCIe 5.0  (x16 link, FL)
    -   Storage
        -   M.2: 1 x M.2 (PCIe Gen5x4, support 2280) (SATA Mode support)
        -   MCIO
            -   6 x MCIO connector (PCIe Gen5x8), support 12 x NVMe drives
            -   2 x MCIO connector (PCIe Gen5x8), support 16 x SATA drives or 4 x NVMe drives
    -   Networking: 2 x SFP28 25Gb/s (Broadcom BCM57414B1KFSBG) +1 x Mgmt LAN
    -   On Board I/O
        -   1 x USB 3.2 Gen1 header (2 port for front panel)
        -   1 x USB 3.2 Gen1 port (1 port Type-A vertical)
        -   1 x Serial port header
        -   6 x FAN header (4-pin)
        -   1 x TPM header
        -   1 x Chassis Intruder header (2-pin)
    -   [Reddit](https://www.reddit.com/r/homelab/comments/1h1iprj/epyc_97x4_genoa_motherboard/) thread:
        >    I use Asus K14PA-U12.
        >   -   Pros:
        >       -   12 RAM slots
        >       -   8 MCIO connectors (PCIe Gen5 x8)
        >       -   3 x PCIe Gen5 x16
        >       -   1 x M.2 (PCIe Gen5 x4)
        >       -   dual 25 Gbps SFP28
        >       -   price is ~700 USD
        >       -   overclocking settings in BIOS
        >       -   relatively compact form factor
        >   -   Cons:
        >       -   they don't plan to release BIOS upgrade for Epyc Turin (I asked)

    [3.7]: https://www.newegg.com/p/pl?N=100007629+601411369&srsltid=AfmBOorDHas0epelrMNy2kXSwLPh7xgASlrIeIDU2XXqA3ZYBGRT9cm3 "Socket SP5 Server Motherboards | Newegg.com"
    [3.12]: https://www.newegg.com/p/pl?N=100007629+601411369 "Socket SP5 Server Motherboards | Newegg.com"
    [3.13]: https://www.reddit.com/r/homelab/comments/1h1iprj "Epyc 97x4 Genoa Motherboard"

</details>
<details><summary>SSD</summary>

-   Supported SSDs may be limited by the motherboard's qualified vendor list (QVL)
-   [Samsung 990 PRO 4TB (MZ-V9P4T0BW)](https://www.techpowerup.com/ssd-specs/samsung-990-pro-4-tb.d863)
-   [MBD-H13SSL-NT compatible SSD list](https://www.supermicro.com/en/support/resources/m2ssd?SystemID=88241&ProductName=H13SSL-NT)
    | Part Number                   | Manufacturer | Manufacturer Part #         | Capacity | Description                                                    |
    |-------------------------------|--------------|-----------------------------|----------|---------------------------------------------------------------|
    | HDS-M2N4-400G0-E3-TXD-NON-080 | Micron       | MTFDKBA400TFS-1BC1ZABYY     | 400GB    | Micron 7450 MAX 400GB NVMe PCIe 4.0 3D TLC M.2 22x80 mm, 3DWPD |
    | HDS-M2N4-480G0-E1-T1E-OSE-080 | Micron       | MTFDKBA480TFR-1BC15ABYY     | 480GB    | Micron 7450 PRO 480GB NVMe PCIe 4.0 M.2 22x80mm TCG Opal 2.0, 1DWPD |
    | HDS-M2N4-480G0-E1-TXD-NON-080 | Micron       | MTFDKBA480TFR-1BC1ZABYY     | 480GB    | Micron 7450 PRO 480GB NVMe PCIe 4.0 M.2 22x80mm 3D TLC, 1DWPD  |
    | HDS-M2N4-800G0-E3-TXD-NON-080 | Micron       | MTFDKBA800TFS-1BC1ZABYY     | 800GB    | Micron 7450 MAX 800GB NVMe PCIe 4.0 M.2 22x80 mm, 3DWPD 3D TLC |
    | HDS-M2N4-960G0-E1-TXE-OSE-080 | Micron       | MTFDKBA960TFR-1BC15ABYY     | 960GB    | Micron 7450 PRO 960GB NVMe PCIe 4.0 M.2 22x80mm TCG Opal 2.0, 1DWPD |
    | HDS-M2N4-960G0-E1-TXD-NON-080 | Micron       | MTFDKBA960TFR-1BC1ZABYY     | 960GB    | Micron 7450 PRO 960GB NVMe PCIe 4.0 M.2 22x80mm 3D TLC, 1DWPD  |
    | HDS-M2N4-001T9-E1-TXE-OSE-110 | Micron       | MTFDKBG1T9TFR-1BC15ABYY     | 1920GB   | Micron 7450 PRO 1.9TB NVMe PCIe 4.0 M.2 22x110mm TCG Opal 2.0, 1DWPD |
    | HDS-M2N4-001T9-E1-TXD-NON-110 | Micron       | MTFDKBG1T9TFR-1BC1ZABYY     | 1920GB   | Micron 7450 PRO 1.9TB NVMe PCIe 4.0 M.2 22x110mm 3D TLC, 1DWPD |
    | HDS-M2N4-003T8-E1-TXD-NON-110 | Micron       | MTFDKBG3T8TFR-1BC1ZABYY     | 3840GB   | Micron 7450 PRO 3.8TB NVMe PCIe 4.0 M.2 22x110mm 3D TLC, 1DWPD |
    | HDS-M2N4-960G0-E1-TXD-NON-110 | Micron       | MTFDKBG960TFR-1BC1ZABYY     | 960GB    | Micron 7450 PRO 960GB NVMe PCIe 4.0 M.2 22x110mm 3D TLC, 1DWPD |
    | HDS-M2N4-01T92-E1-T1D-SED-110 | Samsung      | MZ1L21T9HCLS-00A07          | 1920GB   | Samsung PM9A3 1.9TB NVMe PCIe Gen4 V6 M.2 22x110 (1DWPD) SED   |
    | HDS-M2N4-003T8-E1-TXD-SED-110 | Samsung      | MZ1L23T8HBLA-00A07          | 3840GB   | Samsung PM9A3 3.8TB NVMe PCIe Gen4 V6 M.2 22x110 (1DWPD) SED   |
    | HDS-M2N4-960G0-E1-T1D-SED-110 | Samsung      | MZ1L2960HCJR-00A07          | 960GB    | Samsung PM9A3 960GB NVMe PCIe Gen4 V6 M.2 22x110M (1DWPD) SED  |

</details>
<details><summary>PSU</summary>

-   CPU + RAM + SSD + motherboard: ~300-500 W
-   GPU:
    -   NVIDIA GeForce RTX 5090: 575 W
-   Total for a 2-GPU setup:
    -   500 + 2 x 575 = 1550 W
    -   Headroom: 50%
    -   Required power supply: 2375 W
    -   [Seasonic Prime PX-2200 2200W 80 PLUS Platinum](https://seasonic.com/atx3-prime-px-2200/)
        -   Total continuous power 	2200 W
-   Total for a 4-GPU setup:
    -   500 + 4 x 575 = 2800 W
    -   Headroom: 50%
    -   Required power supply: 4200 W

</details>
<details><summary>Chassis</summary>

-   Review: ([Gamers Nexus - Best PC Cases of 2022 - Best Thermals (Fractal Torrent)](https://youtu.be/pL5uttjPWZE?t=678))
-   [Fractal Design Torrent](https://www.fractal-design.com/products/cases/torrent/torrent/black-solid/)
    -   Expansion slots: 7
    -   Front interface: 1x USB 3.2 Gen 2x2 Type-C (20 Gbps), 2x USB 3.0, HD Audio
    -   Total fan mounts: 7x 120/140 mm or 4x 180 mm
    -   Front fan: 3x 120/140 mm or 2x 180 mm (2x Dynamic GP-18 included in standard version, 2x Prisma AL-18 included in RGB version)
    -   Rear fan: 1x 120/140 mm
    -   Bottom fan: 3x 120/140 mm or 2x 180 mm (3x Dynamic GP-14 PWM included in standard version, 3x Prisma AL-14 PWM included in RGB version)
    -   Dust filters: Front, Bottom
    -   Fixed cable straps: Yes
    -   Cable routing grommets: Yes
    -   Tool-less push-to-lock: Both side panels
    -   Captive thumbscrews: HDD brackets, SSD brackets, Top panel, Bottom fan bracket
    -   Left side panel: Steel or Tempered glass (RGB version: Tempered glass only)
    -   Right side panel: Steel or Tempered glass (Solid/White RGB: Steel, TG/Black RGB: Tempered Glass)
    -   Compatibility:
        -   Motherboard: E-ATX / ATX / mATX / ITX / SSI-EEB / SSI-CEB
        -   Power supply: ATX
        -   PSU max length: 230 mm
        -   GPU max length: 461 mm total, 423 mm with front fan mounted
        -   CPU cooler max height: 188 mm
        -   Front radiator: Up to 360/420 mm, including 360x180 mm
        -   Rear radiator: Up to 120/140 mm
        -   Bottom radiator: Up to 360/420 mm (458 mm max length)
        -   Cable routing space: 32 mm
    -   Dimensions:
        -   Case dimensions (LxWxH): 544 x 242 x 530 mm
        -   Case dimensions w/o feet/protrusions/screws: 525 x 242 x 495 mm
        -   Net weight: 11.1 kg (Solid: 10.4 kg, White TG: 10.8 kg)
        -   Package dimensions (LxWxH): 640 x 343 x 674 mm
        -   Gross weight: 13.7 kg (Solid: 13 kg, White TG: 13.4 kg)
-   [SilverStone SETA H2](https://www.silverstonetek.com/en/product/info/computer-chassis/seh2_b/)
    -   Model No.: SST-SEH2-B
    -   Material: Steel
    -   Motherboard support: SSI-EEB, SSI-CEB, Extended ATX, ATX, Micro-ATX, Mini-ITX
    -   Drive bays:
        -   Internal: 3.5"/2.5" x 11, 3.5" x 1 / 2.5" x 2, 2.5" x 2
    -   Cooling system:
        -   Front: 120mm x 3 / 140mm x 3
        -   Rear: 120mm x 1 / 140mm x 1
        -   Top: 120mm x 3 / 140mm x 3 / 160mm x 2
        -   Side: 120mm x 2
    -   Radiator support:
        -   Front: 120mm / 140mm / 240mm / 280mm / 360mm / 420mm
        -   Rear: 120mm / 140mm
        -   Top: 120mm / 140mm / 240mm / 280mm / 360mm / 420mm
        -   Side: 120mm / 240mm
    -   CPU cooler height limit: 188mm
    -   Expansion slots: 8
    -   Expansion card length limit:
        -   428.9mm (with front 25mm thickness fans installed)
        -   330mm (with side radiator & fans installed)
    -   Power supply: Standard PS2 (ATX)
        -   PSU length limit: 220mm
    -   Front I/O ports:
        -   USB Type-C x 1
        -   USB 3.0 x 2
        -   Combo Audio x 1
    -   Dimensions: 244.9mm (W) x 528.3mm (H) x 543.2mm (D), 70.28 liters
        -   9.64" (W) x 20.8" (H) x 21.39" (D), 70.28 liters
    -   See also [Level1Techs: Our DUAL RTX 5090 Silverstone MADNESS Build: Part 1!](https://www.youtube.com/watch?v=VrTHwN6OKG0)
-   [SuperChassis 747BTQ-R2K04B](https://www.supermicro.com/en/products/chassis/4U/747/SC747BTQ-R2K04B)
    -   8x 3.5” SAS/SATA Backplane for Hot-Swappable Drives (Support SES2)
    -   11x Full-Height, Full-Length Expansion Slots Optimized for 4x Double Width GPU Solution
    -   (2x) Rear Additional 80mm PWM Fans & (4x) Middle Lower 92mm PWM Fans
    -   4U / Full Tower Chassis Supports max. Motherboard, Sizes – E-ATX 15.2” x 13.2”/ ATX/Micro ATX
    -   2000W Redundant Titanium Level Certified High-Efficiency Power Supply
    -   3x 5.25" External HDD Drive Bays & 8x 3.5” Hot-Swappable HDD Drives
    -   Form Factor: 4U tower/rachmount chassis - supports for maximum motherboard sizes: 15.2" x 13.2"
    -   Processor Support: Dual and Single Intel® and AMD processors
    -   Systems Cooling Fans
        -   2x 80mm Hot-swap PWM Fans
        -   4x 92mm hot-swap fan(s)
    -   Power Supply: 1U 2000W Titanium Redundant Power Supply W/PMbus
-   [Fractal Design Define 7](https://www.fractal-design.com/products/cases/define/define-7/)
    -   Total fan mounts: 9 x 120/140 mm
    -   Front fan: 3 x 120/140 mm (2 x Dynamic X2 GP-14 included)
    -   Top fan: 3 x 120/140 mm
    -   Rear fan: 1 x 120/140 mm (1 x Dynamic X2 GP-14 included)
    -   Bottom fan: 2 x 120/140 mm
    -   GPU max length:
        -   Storage layout: 290 mm
        -   Open layout: 470 mm (445 mm w/ front fan)
    -   CPU cooler max height: 185 mm
        -   OK with [Noctua NH-D15 G2](https://noctua.at/en/nh-d15-g2/specification) CPU cooler
    -   Front radiator: Up to 360/280 mm
    -   Top radiator: Up to 360/420 mm
    -   Rear radiator: 120 mm
    -   Vertical GPU Support (with Flex B-20 or Flex VRC-25): 65mm total
        clearance, standard 2-slot GPU (&lt;38mm thickness) recommended for
        optimum cooling
    -   Case dimensions (LxWxH): 547 x 240 x 475 mm
-   A 4U chassis cannot accommodate the [NVIDIA RTX PRO 6000 Blackwell Desktop](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4272)
    version. The card is 137 mm high, but the power connector is located
    at the top. The RTX 4090 is the same height, and the power connector
    is also located on the top. The [Squeezing an RTX 4090 into the 4u Rosewell Server Chassis](https://youtu.be/HQ2EEQkbk8Y?t=289)
    video demonstrates that even though the card itself fits into the
    chassis, the protruding power connector prevents mounting the lid.

</details>
<details><summary>GPU</summary>

-   AM5/X870E platforms split the CPU PEG lanes to x8/x8 for dual GPUs; x16/x16 isn’t available.
-   Physical fit: most RTX 5090 cards are 3–3.5-slot wide. Fitting two often requires:
    -   A case with 8+ expansion slots and generous bottom clearance
    -   Motherboard slot spacing that leaves at least 3 full slots between
        x16_1 and x16_2
-   Power: plan for a high-end PSU (typically 1600–2000W) and adequate
    12V‑2×6 connectors; some boards (e.g., MSI GODLIKE) include a
    supplemental PCIe slot power header that helps stability with dual
    GPUs.
-   No SLI/NVLink for 5090; dual-GPU is for compute (CUDA/ML), not gaming AFR.

</details>

### Prices

<details><summary>CPU prices</summary>


-   AMD EPYC Zen 4 / Zen 5 processors with a TDP of less than 300 W
    -   Zen 4:
        -   Below 300 W: 9354, 9354P, 9454, 9454P, 9534, 9634
        -   [Above 300 W](https://www.amd.com/en/products/specifications/server-processor.html):
            9174F, 9184X, 9274F, 9374F, 9384X, 9474F, 9554, 9554P, 9654, 9654P,
            9684X, 9734, 9754, 9754S

        | Processor Model   | Listed Price (Ft) | Notes / Citation               |
        |-------------------|-------------------|--------------------------------|
        | **Below 300 W**   |                   |                                |
        | 9354 ←            | 1 120 491         | ([Árukereső.hu][cpu_zen4_0_1]) |
        | 9354P             |   917 673         | ([Árukereső.hu][cpu_zen4_0_1]) |
        | 9454              |   913 180         | ([Árukereső.hu][cpu_zen4_0_1]) |
        | 9454P             |   829 448         | ([Árukereső.hu][cpu_zen4_0_1]) |
        | 9534 ←            |   753 930         | ([Árukereső.hu][cpu_zen4_0_2]) |
        | 9634              | 1 963 614         | ([Árukereső.hu][cpu_zen4_0_2]) |
        | **Above 300 W**   |                   |                                |
        | 9174F             | 1 082 453         | ([Árukereső.hu][cpu_zen4_1_1]) |
        | 9184X ←           | 1 660 384         | ([Árukereső.hu][cpu_zen4_1_2]) |
        | 9274F             |   655 800         | ([Árukereső.hu][cpu_zen4_1_1]) |
        | 9374F             |   985 980         | ([Árukereső.hu][cpu_zen4_1_3]) |
        | 9384X ←           | 1 997 695         | ([Árukereső.hu][cpu_zen4_1_2]) |
        | 9474F             | 1 572 490         | ([Árukereső.hu][cpu_zen4_1_1]) |
        | 9554              |   807 950         | ([Árukereső.hu][cpu_zen4_1_1]) |
        | 9554P             | 1 079 486         | ([Árukereső.hu][cpu_zen4_2_1]) |
        | 9654              |   917 690         | ([Árukereső.hu][cpu_zen4_2_2]) |
        | 9654P             |   935 813         | ([Árukereső.hu][cpu_zen4_2_2]) |
        | 9684X ←           | 2 118 660         | ([Árukereső.hu][cpu_zen4_2_3]) |
        | 9734              | 1 021 110         | ([Árukereső.hu][cpu_zen4_2_4]) |
        | 9754 ←            | 2 891 190         | ([Árukereső.hu][cpu_zen4_2_5]) |

        [cpu_zen4_0_1]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9354-32-core-3-25ghz-sp5-tray-100-000000798-p923300802/
        [cpu_zen4_0_2]: https://www.arukereso.hu/processzor-c3139/f%3A5-nm%2Camd-socket-sp5/ "AMD Socket SP5, Gyártási technológia - Processzor - Árukereső.hu"
        [cpu_zen4_1_1]: https://www.arukereso.hu/processzor-c3139/f%3A5-nm%2Camd-socket-sp5/ "AMD Socket SP5, Gyártási technológia - Processzor - Árukereső.hu"
        [cpu_zen4_1_2]: https://www.arukereso.hu/processzor-c3139/f%3Aamd-epyc%2C768-mb-l3-cache/ "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc, L3 ..."
        [cpu_zen4_1_3]: https://www.arukereso.hu/processzor-c3139/f%3Aamd-socket-sp5%2Camd-epyc/ "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc, AMD Socket SP5"
        [cpu_zen4_2_1]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9554p-64-core-3-1ghz-sp5-tray-100-000000804-p924473337/ "AMD EPYC 9554P 64-Core 3.1GHz SP5 Tray (100-000000804 ..."
        [cpu_zen4_2_2]: https://www.arukereso.hu/processzor-c3139/f%3A5-nm%2Camd-socket-sp5/ "AMD Socket SP5, Gyártási technológia - Processzor - Árukereső.hu"
        [cpu_zen4_2_3]: https://www.arukereso.hu/processzor-c3139/f%3A96-magos-processzor%2Camd-socket-sp5/?orderby=13 "Vásárlás: Processzor árak összehasonlítása - AMD ... - Árukereső.hu"
        [cpu_zen4_2_4]: https://www.arukereso.hu/processzor-c3139/f%3A112-magos-processzor%2Camd-epyc/ "Típus: AMD Epyc, 112 magos processzor - Árukereső.hu"
        [cpu_zen4_2_5]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9754-128-core-2-25ghz-sp5-tray-100-000001234-p992134270/
        [cpu_zen4_2_6]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9754s-2-25ghz-tray-p1046085382/ "AMD Epyc 9754S 2.25GHz Tray vásárlás, olcsó ... - Árukereső.hu"

    -   Zen 5:
        -   Below 300 W: 9355P, 9355, 9365, 9455P, 9455, 9535
        -   Above 300 W: 9175F, 9275F, 9375F, 9475F, 9555, 9555P, 9565

        | Processor Model   | Lowest Listed Price (Ft) | Notes / Citation               |
        |-------------------|--------------------------|--------------------------------|
        | **Below 300 W**   |                          |                                |
        | 9355P             | 1 424 902                | ([Árukereső.hu][cpu_zen5_0_1]) |
        | 9355 ←            | 1 170 521                | ([Árukereső.hu][cpu_zen5_0_2]) |
        | 9365              | 1 365 900                | ([Árukereső.hu][cpu_zen5_0_2]) |
        | 9455P             | 1 557 900                | ([Árukereső.hu][cpu_zen5_0_3]) |
        | 9455              | 1 594 330                | ([Árukereső.hu][cpu_zen5_0_4]) |
        | 9535              | 2 369 884                | ([Árukereső.hu][cpu_zen5_0_5]) |
        | **Above 300 W**   |                          |                                |
        | 9175F             | 1 130 275                | ([Árukereső.hu][cpu_zen5_1_1]) |
        | 9275F             | 1 143 398                | ([Árukereső.hu][cpu_zen5_1_2]) |
        | 9375F             | 2 129 697                | ([Árukereső.hu][cpu_zen5_1_3]) |
        | 9475F ←           | 1 694 576                | ([Árukereső.hu][cpu_zen5_1_4]) |
        | 9555              | 2 329 744                | ([Árukereső.hu][cpu_zen5_1_6]) |
        | 9555P             | 2 095 250                | ([Árukereső.hu][cpu_zen5_1_6]) |
        | 9565              | 2 388 790                | ([Árukereső.hu][cpu_zen5_1_7]) |
        | 9575F             | 2 840 784                | ([Árukereső.hu][cpu_zen5_2_1]) |
        | 9645              | 2 770 900                | ([Árukereső.hu][cpu_zen5_2_2]) |
        | 9655              | 2 431 037                | ([Árukereső.hu][cpu_zen5_2_1]) |
        | 9655P             | 3 028 591                | ([Árukereső.hu][cpu_zen5_2_3]) |
        | 9745              | — Not found —            | No listings located            |
        | 9825              | — Not found —            | No listings located            |
        | 9845              | — Not found —            | No listings located            |

        [cpu_zen5_0_1]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9355p-32-core-3-55ghz-sp5-tray-100-000001521-p1149737872/ "AMD EPYC 9355P 32-Core 3.55GHz SP5 Tray (100-000001521 ..."
        [cpu_zen5_0_2]: https://www.arukereso.hu/processzor-c3139/f%3A4-nm%2Camd-epyc/?orderby=13 "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc ..."
        [cpu_zen5_0_3]: https://www.arukereso.hu/processzor-c3139/kiszereles-talcas-oem/ "Kiszerelés: Tálcás (OEM) - Processzor - Árukereső.hu"
        [cpu_zen5_0_4]: https://www.arukereso.hu/processzor-c3139/f%3Aamd-epyc%2Ckiszereles-talcas-oem/ "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc ..."
        [cpu_zen5_0_5]: https://www.arukereso.hu/processzor-c3139/amd-epyc/?st=9535 "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc 9535"
        [cpu_zen5_1_1]: https://www.arukereso.hu/processzor-c3139/512-mb-l3-cache/ "L3 cache: 512 MB - Processzor árak összehasonlítása - Árukereső.hu"
        [cpu_zen5_1_2]: https://www.arukereso.hu/processzor-c3139/f%3Aamd-socket-sp5%2Camd-epyc/ "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc ..."
        [cpu_zen5_1_3]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9375f-32-core-3-8ghz-sp5-tray-100-000001197-p1162090915/ "AMD EPYC 9375F 32-Core 3.8GHz SP5 Tray (100-000001197 ..."
        [cpu_zen5_1_4]: https://www.arukereso.hu/processzor-c3139/f%3Aamd-epyc%2C256-mb-l3-cache/ "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc, L3 ..."
        [cpu_zen5_1_5]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9475f-48-core-3-65ghz-sp5-tray-100-000001143-p1163945305/ "AMD EPYC 9475F 48-Core 3.65GHz SP5 Tray (100-000001143 ..."
        [cpu_zen5_1_6]: https://www.arukereso.hu/processzor-c3139/f%3A64-magos-processzor%2Camd-epyc/ "Típus: AMD Epyc, 64 magos processzor - Árukereső.hu"
        [cpu_zen5_1_7]: https://www.arukereso.hu/processzor-c3139/f%3Aamd-epyc%2Ckiszereles-talcas/?orderby=1&start=75 "Vásárlás: Processzor árak összehasonlítása - Típus: AMD Epyc ..."
        [cpu_zen5_2_1]: https://www.arukereso.hu/processzor-c3139/amd-socket-sp5/ "Processzor árak összehasonlítása - AMD Socket SP5 - Árukereső.hu"
        [cpu_zen5_2_2]: https://www.arukereso.hu/processzor-c3139/96-magos-processzor/ "Vásárlás: Processzor árak összehasonlítása - 96 magos processzor"
        [cpu_zen5_2_3]: https://www.arukereso.hu/processzor-c3139/4-nm/ "Processzor árak összehasonlítása - Gyártási technológia: 4 nm"

</details>
<details><summary>CPU cooler prices</summary>

- [Arctic Freezer 4U-SP5](https://www.arukereso.hu/szamitogep-huto-c3094/arctic/freezer-4u-sp5-acfre00158a-p1161603583/): 70 EUR

</details>
<details><summary>RAM prices</summary>

-   Vendors:
    -   [Micron RDIMM memory part catalog](https://www.micron.com/products/memory/dram-modules/rdimm/part-catalog)
-   Requirement:
    -   [DDR5 RDIMM 1Rx4 or 2Rx8](https://www.arukereso.hu/memoria-modul-c3577/f:kapacitas-32-gb,memoria-tipusa-ddr5,tipus-szerver-memoria,sebesseg=4800-9200/?orderby=1&st=RDIMM)

-   For 1-CPU architecture: 12 x 32 GB RDIMM 1Rx4 or 2Rx8 (384 GB in total)
    -   Candidates:
        -   12 x [Micron RDIMM DDR5 32GB 2Rx8 6400MHz PC5-51200 ECC REGISTERED | MTC20F2085S1RC64BR](https://www.senetic.hu/product/MTC20F2085S1RC64BR): 12 x 180 = 2100 EUR
        -   12 x [Micron 32GB DDR5 4800MHz MTC20F2085S1RC48BR](https://www.arukereso.hu/memoria-modul-c3577/micron/32gb-ddr5-4800mhz-mtc20f2085s1rc48br-p1004252032/) 12 x 200 = 2400 EUR
        -   12 x [Samsung 32GB DDR5 5600MHz M321R4GA3PB0-CWM](https://www.arukereso.hu/memoria-modul-c3577/samsung/32gb-ddr5-5600mhz-m321r4ga3pb0-cwm-p1051234201/): 12 x 200 = 2400 EUR
        -   12 x [M321R4GA0BB0-CQK](https://semiconductor.samsung.com/dram/module/rdimm/m321r4ga0bb0-cqk/),
            [Árukereső](https://www.arukereso.hu/memoria-modul-c3577/samsung/32gb-ddr5-4800mhz-m321r4ga0bb0-cqk-p921898419/): 12 x 316 = 3800 EUR
        -   12 x [M321R4GA3BB6-CQK](https://semiconductor.samsung.com/dram/module/rdimm/m321r4ga3bb6-cqk/),
            [Árukereső](https://www.arukereso.hu/memoria-modul-c3577/samsung/32gb-ddr5-4800mhz-m321r4ga3bb6-cqk-p872096736/): 12 x 255 = 3000 EUR
        -   12 x [M329R4GA0BB0-CQK](https://semiconductor.samsung.com/dram/module/rdimm/m329r4ga0bb0-cqk/)
        -   Kingston
            | Memory Module            | Price (HUF)            | Notes / Source                                         |
            | ------------------------ | ---------------------- | ------------------------------------------------------ |
            | **KSM48R40BD8-32HA**     | from 77 175 Ft         | Árukereső listings ([Árukereső.hu][ram_kingston_1])                 |
            | **KSM48R40BD8-32MD**     | from 77 175 Ft         | Same listing covers both HA and MD ([Árukereső.hu][ram_kingston_1]) |
            | **KSM56R46BD8-32MD**     | 76 590 Ft              | Direct offer ([Árukereső.hu][ram_kingston_2])                       |
            | **KSM56R46BD8PMI-32MDI** | from 93 900 Ft         | Árukereső comparison ([Árukereső.hu][ram_kingston_3])               |
            | **KSM64R52BD8-32MD**     | 161 890 Ft             | Direct listing ([Árukereső.hu][ram_kingston_4])                     |

            [ram_kingston_1]: https://www.arukereso.hu/memoria-modul-c3577/f%3Akingston%2Ctipus-szerver-memoria/?start=75 "Vásárlás: Kingston Memória modul árak összehasonlítása - Típus"
            [ram_kingston_2]: https://www.arukereso.hu/memoria-modul-c3577/kingston/32gb-ddr5-5600mhz-ksm56r46bd8-32md-p1128575413/ "Kingston 32GB DDR5 5600MHz KSM56R46BD8 ... - Árukereső.hu"
            [ram_kingston_3]: https://www.arukereso.hu/memoria-modul-c3577/f%3Akingston%2Cmemoriakesleltetes-cl-46/?orderby=1 "Vásárlás: Kingston Memória modul árak ... - Árukereső.hu"
            [ram_kingston_4]: https://www.arukereso.hu/memoria-modul-c3577/kingston/32gb-ddr5-5200mhz-ksm64r52bd8-32md-p1190480737/ "Kingston 32GB DDR5 5200MHz KSM64R52BD8-32MD memória ..."
        -   Samsung
            | Memory Module        | Price (HUF)         | Notes                                       |
            |----------------------|---------------------|---------------------------------------------|
            | **M321R4GA0BB0-CQK** | from **126 105 Ft** | Listed on Árukereső ([Árukereső.hu][ram_samsung_1])  |
            | **M321R4GA3BB6-CQK** | from **91 900 Ft**  | Listed on Árukereső ([Árukereső.hu][ram_samsung_1])  |
            | **M321R4GA3EB0-CWM** | from **78 272 Ft**  | Listed on Árukereső ([Árukereső.hu][ram_samsung_2])  |
            | **M321R4GA3PB0-CWM** | from **74 580 Ft**  | Listed on Árukereső ([Árukereső.hu][ram_samsung_3])  |

            [ram_samsung_1]: https://www.arukereso.hu/memoria-modul-c3577/f%3Asamsung%2Cmemoria-tipusa-ddr5/?orderby=1 "Olcsó DDR5 Samsung memória - Árukereső.hu"
            [ram_samsung_2]: https://www.arukereso.hu/memoria-modul-c3577/samsung/32gb-ddr5-5600mhz-m321r4ga3eb0-cwm-p1194855955/ "Samsung 32GB DDR5 5600MHz M321R4GA3EB0-CWM memória ..."
            [ram_samsung_3]: https://www.arukereso.hu/memoria-modul-c3577/f%3Asamsung%2Cmemoriakesleltetes-cl-46/ "Vásárlás: Samsung Memória modul árak ... - Árukereső.hu"
        -   Micron
            | Memory Module                                 | Price (HUF)                                       | Availability |
            | --------------------------------------------- | ------------------------------------------------- | ------------ |
            | **MTC20F2085S1RC48BR** (32 GB, DDR5 4800 MHz) | from **82 648 Ft** ([Árukereső.hu][ram_micron_1]) | Available    |
            | **MTC20F2085S1RC56BR** (32 GB, DDR5 5600 MHz) | from **69 602 Ft** ([Árukereső.hu][ram_micron_2]) | Available    |

            [ram_micron_1]: https://www.arukereso.hu/memoria-modul-c3577/f%3Amicron%2Ckapacitas-32-gb/ "Vásárlás: Micron Memória modul árak összehasonlítása - Kapacitás ..."
            [ram_micron_2]: https://www.arukereso.hu/memoria-modul-c3577/f%3Amicron%2Cmemoria-tipusa-ddr5/ "Micron Memória modul árak összehasonlítása - DDR5 - Árukereső.hu"
-   For 2-CPU architecture:
    -   24 x [Kingston 16GB DDR5 4800MHz KSM48E40BS8KI-16HA](https://www.arukereso.hu/memoria-modul-c3577/kingston/16gb-ddr5-4800mhz-ksm48e40bs8ki-16ha-p1054408474/): 24 x 100 = 2400 EUR

</details>
<details><summary>SSD prices</summary>

-   [Samsung 990 PRO 4TB (MZ-V9P4T0BW)](https://belso-ssd-meghajto.arukereso.hu/samsung/990-pro-4tb-mz-v9p4t0bw-p1002242350/): 300 EUR

</details>
<details><summary>Motherboard prices</summary>

-   CEB:
    -   [Asus K14PA-U12](https://smicro.hu/asus-k14pa-u12-90sb0ci0-m0uay0-4?aku=db3621a52f6055ee636a6fee6ff8a353): 800 EUR
-   ATX:
    -   [Supermicro MBD-H13SSL-NT-O](https://smicro.hu/supermicro-mbd-h13ssl-nt-o-4): 830 EUR
    -   [GIGABYTE MZ33-AR0](https://www.arukereso.hu/alaplap-c3128/gigabyte/mz33-ar0-p1005435430/): 1100 EUR

</details>
<details><summary>PSU prices</summary>

-   [Seasonic Prime PX-2200 2200W 80 PLUS Platinum](https://www.arukereso.hu/tapegyseg-c3158/seasonic/prime-px-2200-2200w-80-plus-platinum-p1129871905/): 630 EUR

</details>
<details><summary>Chassis prices</summary>

-   [Fractal Design Torrent](https://www.arukereso.hu/szamitogep-haz-c3085/f:fractal-design,szelesseg=4/?orderby=1&st=torrent): 240 EUR
-   [SilverStone H1 SST-SEH1B-G](https://www.arukereso.hu/szamitogep-haz-c3085/silverstone/h1-sst-seh1b-g-p853480299/): 210 EUR

</details>
<details><summary>Vendor sites</summary>

-   https://smicro.hu/amd-socket-sp5-5
-   https://www.senetic.hu/category/amd-cpu-epyc-9004-11151/
-   https://www.arukereso.hu/processzor-c3139/f:tdp=0-350,amd-socket-sp5,amd-epyc/?orderby=1
-   Motherboards:
    -   https://smicro.hu/amd-sp5-5?filtrPriceFrom=&filtrPriceTo=&filter%5B2294%5D%5B%5D=39137&filter%5B2424%5D%5B%5D=42927&filter%5B2317%5D%5B%5D=38124&filter%5B2316%5D%5B%5D=38705&filter%5B2316%5D%5B%5D=39193&filter%5B2315%5D%5B%5D=40251&filter%5B2315%5D%5B%5D=43437&filter%5B2360%5D%5B%5D=39223
-   https://geizhals.eu/

</details>

### Links

<details><summary>Links</summary>

-   [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1gcn3w9/a_glance_inside_the_tinybox_pro_8_x_rtx_4090/) on building an 8x 4090 configuration
    >   I've built much the same thing here with 8x 4090, only mine lives in
    >   an open air mining frame I designed and I used ROME2D32GM-2T
    >   motherboard as I didn't see any point in Genoa when none of the cards
    >   can use PCIe Gen5. My configuration is:
    >   -   Mobo: ASRockRack ROME2D32GM-2T
    >   -   CPU: 2x AMD Epyc 7443
    >   -   CPU Cooler: 2x Noctua NH-U14S TR4-SP3
    >   -   Memory: 8x Samsung M393A4K40EB3-CWE
    >   -   GPU: 8x MSI GeForce RTX 4090 Gaming X Slim
    >   -   GPU adapters: 8x C-Payne SlimSAS PCIe gen4 Device Adapter x8/x16
    >   -   GPU cable set 1: 8x C-Payne SlimSAS SFF-8654 8i cable - PCIe gen4
    >   -   GPU cable set 2: 8x C-Payne SlimSAS SFF-8654 to SFF-8654LP (Low Profile) 8i cable - PCIe gen4
    >   -   PSU: 4x Thermaltake Toughpower GF3 1650W
    >   -   Boot drive: Samsung SSD 990 PRO 2TB, M.2 2280
    >   -   Data drives: 4x Samsung SSD 990 PRO 4TB, M.2 2280
    >   -   Data drive adapter: C-Payne SlimSAS PCIe gen4 Device Adapter x8/x16
    >   -   Data drive breakout: EZDIY-FAB Quad M.2 PCIe 4.0/3.0 X16 Expansion Card with Heatsink
    >   -   Data drive cable set: 2x C-Payne SlimSAS SFF-8654 to SFF-8654LP (Low Profile) 8i cable - PCIe gen4
    >   -   Case: Custom open air miner frame built from 2020 alu extrusions
-   [Smallest RTX Pro 6000 rig | OVERKILL](https://www.youtube.com/watch?v=JbnBt_Aytd0)
    >   -   Low profile (quiet) Noctua fans: https://amzn.to/4nIXWM4
    >   -   Water CPU cooler: https://amzn.to/4nDQUbj
    >   -   New tiny case NR200P V3: https://amzn.to/4lnMROM
    >   -   Extremely fast NVMe SSD: https://amzn.to/4kzaCCn
    >   -   AMD CPU: https://amzn.to/3IkD7pV
    >   -   Fast RAM: https://amzn.to/40Dyr4z
    >   -   Mini-ITX Motherboard with EVERYTHING!: https://amzn.to/4eIVgd6
    >   -   Tiny 1000W power supply: https://amzn.to/4lF471z
    >   -   RTX Pro 6000 GPU: https://amzn.to/4eIUnRZ
-   [Building an Efficient GPU Server with NVIDIA GeForce RTX 4090s/5090s](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)
    >   -   Server model: ASUS ESC8000A-E12P
    >   -   GPUs: 8x NVIDIA RTX 4090
    >   -   CPU: 2x AMD EPYC 9254 Processor (24-core, 2.90GHz, 128MB Cache)
    >   -   RAM: 24x 16GB PC5-38400 4800MHz DDR5 ECC RDIMM (384GB total)
    >   -   Storage: 1.92TB Micron 7450 PRO Series M.2 PCIe 4.0 x4 NVMe SSD (110mm)
    >   -   Operating system: Ubuntu Linux 22.04 LTS Server Edition (64-bit)
    >   -   Networking: 2 x 10GbE LAN ports (RJ45, X710-AT2), one utilized at 10Gb
    >   -   Additional PCIe 5.0 card: ASUS 90SC0M60-M0XBN0


</details> <!-- Server links -->

<div class="page"/>

## Light server

Light server configuration
-   Supports 2x NVIDIA RTX PRO 6000 Blackwell (96GB) Desktop GPUs with open-air cooling
-   Theoretical maximum RAM bandwidth of 115.2 GB/s
-   Suggested processors: AMD EPYC 8224P, 8124P (8024P if 67.6 GB/s memory bandwidth is tolerable)
-   Suggested motherboards: ASRock Rack SIENAD8-2L2T.
    -   Maybe GIGABYTE ME03-PE0 if using one x16 slot at PCIe 4.0.
    -   Maybe ASUS S14NA-U12 if using MCIO extension.

| Component   | Model                                                            | Price each<br>[HUF] | Price subtotal<br>[HUF] |
|-------------|------------------------------------------------------------------|--------------------:|------------------------:|
| CPU         | AMD EPYC [8224P][pr_8224P] (Zen 4)                               |         402,090     |             402,090     |
| RAM         | Micron 64GB DDR5 4800MHz [MTC40F2046S1RC48BA1R][ram_ls_64gb] × 6 |         163,690     |             982,140     |
| SSD         | Samsung PM9A3 1.9TB NVMe PCIe Gen4 V6 M.2 22x110                 |         129,910     |             259,820     |
| Motherboard | ASRock Rack [SIENAD8-2L2T][mbd_SIENAD8-2L2T]                     |         317,034     |             317,034     |
| CPU cooler  | BE QUIET! Silent Loop 3 360mm Liquid cooler                      |          60,960     |              60,960     |
| PSU         | Seasonic Prime PX-2200 2200W 80 PLUS Platinum                    |         215,900     |             215,900     |
| Chassis     | Corsair iCUE 9000D RGB Airflow Big-Tower                         |         266,700     |             266,700     |
| Fans        | Noctua NF-F12 iPPC-3000 Industrial PWM 120mm × 13                |          13,208     |             171,704     |
| **Total**   |                                                                  |                     |         **2,676,348**   |


[pr_8224P]: https://www.arukereso.hu/processzor-c3139/amd/epyc-8224p-24-core-2-55ghz-sp6-tray-100-000001134-p1035460933/
[mbd_SIENAD8-2L2T]: https://www.senetic.hu/product/SIENAD8-2L2T
[ram_ls_64gb]: https://www.arukereso.hu/memoria-modul-c3577/micron/64gb-ddr5-4800mhz-mtc40f2046s1rc48ba1r-p943393176/
[sp6_clr_noctua]: https://ipon.hu/shop/termek/noctua-nh-d9-tr5-sp6-4u-cpu-cooler/2251012?aku=27dcaa5a946a5d25ecbc2b5ca46149b2

### Specs

<details><summary>CPU: AMD EPYC 8004 (SP6 socket)</summary>

-   Minimum 6 x DDR5 4800 MT/s RAM
    -   RAM module side memory bandwidth:
        -   [1-CPU config](https://hothardware.com/Image/Resize/?width=1170&height=1170&imageFile=/contentimages/Article/3257/content/big_epyc-cpu-memory-capabilities.jpg):
            6 x 8 x 4.8 GT/s = 230.4 GB/s
    -   CPU side memory bandwidth:
        -   Zen 4:
            -   4 CCD x 32 x 1.8 GHz = 230.4 GB/s
            -   2 CCD x 32 x 1.8 GHz = 115.2 GB/s
            -   1 CCD x 32 x 1.8 GHz = 67.6 GB/s
-   [Zen 4](https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)):
    -   [96 PCIe 5.0 lanes](https://www.techpowerup.com/cpu-specs/epyc-8224p.c3292#gallery-2) (5 x PCIe 5.0 x16 + 2x PCIe 5.0 x8)
-   [CPU candidates](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series.html#tabs-4380fde236-item-a9a09e8dc2-tab) which support PCIe 5.0 x16:
    -   [AMD EPYC 8004 series processors](https://www.amd.com/content/dam/amd/en/documents/products/epyc/epyc-9008-series-processors-data-sheet.pdf) \
        ![EPYC 9008 Series processors](<assets/AMD EPYC 8004 series processors.png>)
    -   1 CCD, 67.6 GB/s: 8024P
    -   2 CCDs, 115.2 GB/s: 8124P, 8224P
    -   4 CCDs, 230.4 GB/s: 8324P, 8434P, 8534P
-   Suggested picks: AMD EPYC 8224P, 8324P

</details>

<details><summary>Motherboard</summary>

-   [ASRock Rack SIENAD8-2L2T](https://www.asrockrack.com/general/productdetail.asp?Model=SIENAD8-2L2T#Specifications)
    ([Review](https://www.servethehome.com/asrock-rack-sienad8-2l2t-amd-epyc-8004-siena-motherboard-review/))
    -   8 DIMM slots (2DPC/1DPC), supports DDR5 RDIMM
    -   3 PCIe5.0 / CXL1.1 x16, 1 PCIe5.0 x16, 1 PCIe5.0 x8
    -   2 MCIO (PCIe5.0 x8)
    -   2 M.2 (PCIe5.0 x4)
    -   [Memory QVL](https://www.asrockrack.com/general/productdetail.asp?Model=SIENAD8-2L2T#Memory)
        | Type | Speed | DIMM  | Size  | Vendor  | Module                        | Part No         | Cell             |
        |------|-------|-------|-------|---------|-------------------------------|-----------------|------------------|
        | DDR5 | 5600  | RDIMM | 128GB | Micron  | MTC40F2047S1RC56BB1 QLFF      | 4FB7DD8GDF      | Micron           |
        | DDR5 | 4800  | RDIMM | 96GB  | Micron  | MTC40F204WS1RC48BB1 IGFF      | 3LB75D8DHL      | Micron           |
        | DDR5 | 4800  | RDIMM | 96GB  | Samsung | M321RYGA0BB0-CQKZJ            | K4RHE046VB BCQK | Sec              |
        | DDR5 | 4800  | RDIMM | 64GB  | Micron  | MTC40F2046S1RC48BA1 GCCH      | IQA45D8BNH      | Micron           |
        | DDR5 | 4800  | RDIMM | 64GB  | Micron  | MTC40F2046S1RC48BA1 FICC      | 3HA45D8BNH      | Micron           |
        | DDR5 | 4800  | RDIMM | 64GB  | SMART   | SR8G8RD5445-SB                | K4RAH046VB BCQK | Sec              |
        | DDR5 | 4800  | RDIMM | 32GB  | Micron  | MTC20F2085S1RC48BA1 NGCC      | 3FA45D8BNJ      | Micron           |
        | DDR5 | 4800  | RDIMM | 32GB  | Micron  | MTC20F1045S1RC48BA2 HCCH      | IQA45D8BNH      | Micron           |
        | DDR5 | 4800  | RDIMM | 32GB  | Samsung | M321R4GA0BB0-CQKET            | K4RAH046VB BCQK | Sec              |
        | DDR5 | 4800  | RDIMM | 16GB  | SMART   | SR2G8RD5285-SB                | K4RAH086VB FCQK | Sec              |
    -   **This motherboard is a viable choice**
-   [GIGABYTE ME03-PE0](https://www.gigabyte.com/Enterprise/Server-Motherboard/ME03-PE0-rev-1x)
    -   3 x PCIe Gen5 x16 expansion slots
    -   4 x PCIe Gen4 x16 and x8 expansion slots
    -   **Not optimal: Need to use one PCIe 5.0 x16 and one PCIe 4.0 x16 slot**
    -   [Qualified Vendor List](https://download.gigabyte.com/FileList/QVL/server_mb_qvl_ME03-PE0_v1.0.pdf?v=49453fe0c587e648d48438af52747fed)
-   [ASRock Rack SIENAD8UD3](https://www.asrockrack.com/general/productdetail.asp?Model=SIENAD8UD3#Specifications)
    -   2 PCIe5.0 / CXL1.1 x16
    -   **Not suitable: PCIe slots are too close to each other**
-   [ASRock Rack SIENAD8UD2-2Q](https://www.asrockrack.com/general/productdetail.asp?Model=SIENAD8UD2-2Q#Specifications)
    -   2 PCIe5.0 / CXL1.1 x16, 1 PCIe5.0 / CXL1.1 x8
    -   **Not suitable: PCIe slots are too close to each other**
-   [ASRock Rack SIENAD8UD-2L2Q](https://www.asrockrack.com/general/productdetail.asp?Model=SIENAD8UD-2L2Q#Specifications)
    -   2 PCIe5.0 / CXL1.1 x16, 1 PCIe5.0 / CXL1.1 x8,
    -   **Not suitable: PCIe slots are too close to each other**
-   [ASUS S14NA-U12](https://servers.asus.com/products/servers/server-motherboards/S14NA-U12)
    -   2 x PCIe 5.0 x 16 slot (x16 link, FL)
    -   1 x PCIe 5.0 x 8 slot (x8 link, FL)
    -   **Not suitable: PCIe slots are too close to each other**
-   [Advantech ASMB-561](https://www.advantech.com/en-eu/products/1f591987-697f-49f8-9fb0-0cca6b3e01eb/asmb-561/mod_57f454b0-03f1-4f7e-a1f6-a60a0176d7df)
    -   Four PCIe Gen5 x16 slots with CXL support on slot 4/6
    -   **Possible candidate**

</details> <!-- Motherboard -->

<details><summary>CPU cooler</summary>

-   [Noctua NH-D9 TR5-SP6 4U](https://noctua.at/en/nh-d9-tr5-sp6-4u)
    -   [Specification](https://noctua.at/en/nh-d9-tr5-sp6-4u/specification)
    -   [CPU compatibility](https://ncc.noctua.at/cpus/model/AMD-Epyc-8324P-1779): OK
    -   The NH-D9’s direction of airflow is parallel to the long axis of the
        socket, so it is ideal for builds where the hot air is exhausted this
        way.
    -   Height (with fan): 134 mm
    -   Max. airflow: 96,3 m³/h
    -   Max. acoustical noise: 30,6 dB(A)

</details>

<details><summary>Chassis</summary>

-   Review: ([Gamers Nexus - Best PC Cases of 2022 - Best Thermals (Fractal Torrent)](https://youtu.be/pL5uttjPWZE?t=678))
-   [Fractal Design Torrent](https://www.fractal-design.com/products/cases/torrent/torrent/black-solid/)
    -   Expansion slots: 7
    -   Front interface: 1x USB 3.2 Gen 2x2 Type-C (20 Gbps), 2x USB 3.0, HD Audio
    -   Total fan mounts: 7x 120/140 mm or 4x 180 mm
    -   Front fan: 3x 120/140 mm or 2x 180 mm (2x Dynamic GP-18 included in standard version, 2x Prisma AL-18 included in RGB version)
    -   Rear fan: 1x 120/140 mm
    -   Bottom fan: 3x 120/140 mm or 2x 180 mm (3x Dynamic GP-14 PWM included in standard version, 3x Prisma AL-14 PWM included in RGB version)
    -   Dust filters: Front, Bottom
    -   Fixed cable straps: Yes
    -   Cable routing grommets: Yes
    -   Tool-less push-to-lock: Both side panels
    -   Captive thumbscrews: HDD brackets, SSD brackets, Top panel, Bottom fan bracket
    -   Left side panel: Steel or Tempered glass (RGB version: Tempered glass only)
    -   Right side panel: Steel or Tempered glass (Solid/White RGB: Steel, TG/Black RGB: Tempered Glass)
    -   Compatibility:
        -   Motherboard: E-ATX / ATX / mATX / ITX / SSI-EEB / SSI-CEB
        -   Power supply: ATX
        -   PSU max length: 230 mm
        -   GPU max length: 461 mm total, 423 mm with front fan mounted
        -   CPU cooler max height: 188 mm
        -   Front radiator: Up to 360/420 mm, including 360x180 mm
        -   Rear radiator: Up to 120/140 mm
        -   Bottom radiator: Up to 360/420 mm (458 mm max length)
        -   Cable routing space: 32 mm
    -   Dimensions:
        -   Case dimensions (LxWxH): 544 x 242 x 530 mm
        -   Case dimensions w/o feet/protrusions/screws: 525 x 242 x 495 mm
        -   Net weight: 11.1 kg (Solid: 10.4 kg, White TG: 10.8 kg)
        -   Package dimensions (LxWxH): 640 x 343 x 674 mm
        -   Gross weight: 13.7 kg (Solid: 13 kg, White TG: 13.4 kg)
</details>

</details> <!-- Light server Server -->

## Workstation

Workstation configuration
-   Supports 2x NVIDIA RTX PRO 6000 Blackwell (96GB) Desktop GPUs with open-air cooling
-   Theoretical maximum RAM bandwidth of 204.8 GB/s
-   Suggested processors: AMD Ryzen Threadripper 9960X, 7960X

| Component     | Model                                                                    | Price each<br>[HUF] | Price subtotal<br>[HUF] |
|---------------|--------------------------------------------------------------------------|--------------------:|------------------------:|
| CPU           | AMD Ryzen Threadripper [9960X][cpu_tr_9960x] (Zen 5)                     |         718,790     |             718,790     |
| RAM           | KSM64R52BD8-32HA, 32GB 6400MT/s DDR5 ECC Reg CL52 DIMM 2Rx8 Hynix A × 8  |          95,250     |             762,000     |
| SSD           | Samsung 9100 PRO, PCIe 5.0, NVMe 2.0, 2TB M.2 SSD × 2                    |         109,220     |             218,440     |
| Motherboard   | Gigabyte MB Sc Sc sTR5 TRX50 AI TOP, AMD TRX50, 8xDDR5, WI-FI, E-ATX     |         410,210     |             410,210     |
| CPU cooler    | BE QUIET! Silent Loop 3 360mm Liquid cooler                              |          60,960     |              60,960     |
| PSU           | Seasonic Prime PX-2200 2200W 80 PLUS Platinum                            |         215,900     |             215,900     |
| Chassis       | Corsair iCUE 9000D RGB Airflow Big-Tower                                 |         266,700     |             266,700     |
| Fans          | Noctua NF-F12 iPPC-3000 Industrial PWM 120mm × 13                        |          13,208     |             171,704     |
| **Total**     |                                                                          |                     |         **2,824,604**   |

### Specs

<details><summary>CPU: AMD Ryzen Threadripper / Threadripper Pro (sTR5 socket)</summary>

-   PCIe 5.0 lanes:
    -   Required: 2x PCIe 5.0 x16 = 32
    -   Zen 4 provides 48 PCIe 5.0 lanes, Zen 5 128 PCIe 5.0 lanes
        -   Either is fine
-   Zen 4 Threadripper:
    -   RAM module side memory bandwidth: 4 x 8 x 5.2 GT/s = 166.4 GB/s
    -   CPU side memory bandwidth: 4 CCD x 32 x 1.8 GHz = 230.4 GB/s
        -   Minimum 4 CCD is required to saturate the RAM
    -   CPU candidates: AMD Ryzen Threadripper 7960X, 7970X, 7980X
-   Zen 4 Threadripper Pro:
    -   RAM module side memory bandwidth: 8 x 8 x 5.2 GT/s = 332.8 GB/s
    -   CPU side memory bandwidth: 8 CCD x 32 x 1.8 GHz = 460.8 GB/s
        -   Minimum 8 CCD is required to saturate the RAM
    -   CPU candidates: AMD Ryzen Threadripper Pro 7985WX, 7995WX
-   Zen 5 Threadripper:
    -   RAM module side memory bandwidth: 4 x 8 x 6.4 GT/s = 204.8 GB/s
    -   CPU side memory bandwidth: 4 CCD x 32 x 2.0 GHz = 256 GB/s
        -   Minimum 4 CCD is required to saturate the RAM
    -   CPU candidates: AMD Ryzen Threadripper Pro 9960X, 9970X, 9980X
-   Zen 5 Threadripper Pro:
    -   RAM module side memory bandwidth: 8 x 8 x 6.4 GT/s = 409.6 GB/s
    -   CPU side memory bandwidth: 8 CCD x 32 x 2.0 GHz = 512 GB/s
        -   Minimum 8 CCD is required to saturate the RAM
    -   CPU candidates: AMD Ryzen Threadripper Pro 9985WX, 9995WX

-   AMD Ryzen Threadripper CPU lineup
    | Arch   | Branding           | Model    | Cores | L3 cache |
    |--------|--------------------|----------|-------|----------|
    | Zen 5  | Threadripper Pro   | 9995WX   | 96    | 384 MB   |
    | Zen 5  | Threadripper Pro   | 9985WX   | 64    | 256 MB   |
    | Zen 5  | Threadripper Pro   | 9975WX   | 32    | 128 MB   |
    | Zen 5  | Threadripper Pro   | 9965WX   | 24    | 128 MB   |
    | Zen 5  | Threadripper Pro   | 9955WX   | 16    | 64 MB    |
    | Zen 5  | Threadripper Pro   | 9945WX   | 12    | 64 MB    |
    | Zen 5  | Threadripper       | 9980X    | 64    | 256 MB   |
    | Zen 5  | Threadripper       | 9970X    | 32    | 128 MB   |
    | Zen 5  | Threadripper       | 9960X    | 24    | 128 MB   |
    ||||||
    | Zen 4  | Threadripper Pro   | 7995WX   | 96    | 384 MB   |
    | Zen 4  | Threadripper Pro   | 7985WX   | 64    | 256 MB   |
    | Zen 4  | Threadripper Pro   | 7975WX   | 32    | 128 MB   |
    | Zen 4  | Threadripper Pro   | 7965WX   | 24    | 128 MB   |
    | Zen 4  | Threadripper Pro   | 7955WX   | 16    | 64 MB    |
    | Zen 4  | Threadripper Pro   | 7945WX   | 12    | 64 MB    |
    | Zen 4  | Threadripper       | 7980X    | 64    | 256 MB   |
    | Zen 4  | Threadripper       | 7970X    | 32    | 128 MB   |
    | Zen 4  | Threadripper       | 7960X    | 24    | 128 MB   |

-   Theoretical maximum memory bandwidth
    -   The TRX50 motherboards supports both Threadripper and Threadripper Pro
        CPUs, but only 4 memory channels. (Even if 8 RDIMM modules are
        installed, they operate in a 4-channel configuration.)
    -   The WRX90 motherboards support 8-channel memory configuration, but
        they support only Threadripper Pro processors. The Threadripper line
        is not supported.
    -   This table assumes that a TRX50 motherboard is used, and calculates
        with 4 memory channels even for the Threadripper Pro line.

    | Model    | CCDs | CPU BW<br>[GB/s] | Mem ch | Mem speed | Mem BW<br>[GB/s] | OC Mem speed    | OC Mem BW   | Price |
    |----------|------|------------------|--------|-----------|------------------|-----------------|-------------|------:|
    | 9995WX   | 12   | 768.0            | 4(8)   | 6400 MT/s | 204.8            | 8000 MT/s       | 256.0       |       |
    | 9985WX   | 8    | 512.0            | 4(8)   | 6400 MT/s | 204.8            | 8000 MT/s       | 256.0       | 3 417 990 Ft |
    | 9975WX   | 4    | 256.0            | 4(8)   | 6400 MT/s | 204.8            | 7200 MT/s       | 230.4       | 1 754 389 Ft |
    | 9965WX   | 4    | 256.0            | 4(8)   | 6400 MT/s | 204.8            | 7200 MT/s       | 230.4       | 1 230 191 Ft |
    | 9955WX   | 2    | 128.0            | 4(8)   | 6400 MT/s | 204.8            | 7200 MT/s       | 230.4       |       |
    | 9945WX   | 2    | 128.0            | 4(8)   | 6400 MT/s | 204.8            | 7200 MT/s       | 230.4       |       |
    | 9980X    | 8    | 512.0            | 4      | 6400 MT/s | 204.8            | 8000 MT/s       | 256.0       | 2 141 790 Ft |
    | 9970X    | 4    | 256.0 ✅        | 4      | 6400 MT/s | 204.8            | 8000 MT/s ✅    | 256.0 ✅   | [1 157 190 Ft][cpu_tr_9970x] |
    | 9960X    | 4    | 256.0 ✅        | 4      | 6400 MT/s | 204.8            | 8000 MT/s ✅    | 256.0 ✅   | [718 790 Ft][cpu_tr_9960x] |
    ||||||||||
    | 7995WX   | 12   | 499.2            | 4(8)   | 5200 MT/s | 332.8            | 8000 MT/s       | 256.0       |       |
    | 7985WX   | 8    | 460.8            | 4(8)   | 5200 MT/s | 332.8            | 7200 MT/s       | 230.4       | 3 111 630 Ft |
    | 7975WX   | 4    | 230.4            | 4(8)   | 5200 MT/s | 332.8            | 7200 MT/s       | 230.4       | 1 663 189 Ft |
    | 7965WX   | 4    | 230.4            | 4(8)   | 5200 MT/s | 332.8            | 7200 MT/s       | 230.4       | 1 072 890 Ft |
    | 7955WX   | 2    | 115.2            | 4(8)   | 5200 MT/s | 332.8            | 7200 MT/s       | 230.4       |       |
    | 7945WX   | 2    | 115.2            | 4(8)   | 5200 MT/s | 332.8            | 7200 MT/s       | 230.4       |       |
    | 7980X    | 8    | 460.8            | 4      | 5200 MT/s | 166.4            | 8000 MT/s       | 256.0       |       |
    | 7970X    | 4    | 230.4 ✅        | 4      | 5200 MT/s | 166.4            | 7200 MT/s ✅    | 230.4 ✅   | [1 070 690 Ft][cpu_tr_7970x] |
    | 7960X    | 4    | 230.4 ✅        | 4      | 5200 MT/s | 166.4            | 7200 MT/s ✅    | 230.4 ✅   | [621 980 Ft][cpu_tr_7960x] |

    [cpu_tr_7960x]: https://www.arukereso.hu/processzor-c3139/amd/ryzen-threadripper-7960x-24-core-4-2ghz-sp6-str5-box-100-100001352wof-p1026372205/
    [cpu_tr_7970x]: https://www.arukereso.hu/processzor-c3139/amd/ryzen-threadripper-7970x-32-core-4-0ghz-str5-box-100-100001351wof-p1026372193/
    [cpu_tr_9960x]: https://www.arukereso.hu/processzor-c3139/amd/ryzen-threadripper-pro-9960x-24-core-5-4ghz-str5-box-100-100001595wof-p1215565084/
    [cpu_tr_9970x]: https://www.arukereso.hu/processzor-c3139/amd/ryzen-threadripper-pro-9970x-32-core-5-4ghz-str5-box-100-100001594wof-p1215566377/

</details>

<details><summary>RAM</summary>

-   RAM module candidates for the Threadripper 9000 series CPUs, 4 x 64 = 256 GB configuration:
    | Speed | Supplier    | Capacity | Rank     | Module P/N            | Chip Brand | Timing           | Voltage | Native |
    |-------|-------------|----------|----------|-----------------------|------------|------------------|---------|--------|
    | 7200  | V-COLOR     | 64GB     | 2Rx8     | TRA564G72D836         | Hynix M    | 36-51-51-112     | 1.4V    | 6400   |
    | 7200  | V-COLOR     | 64GB     | 2Rx8     | TRA564G72D836Q ✅     | Hynix M    | 36-51-51-112     | 1.4V    | 6400   |
    | 7200  | V-COLOR     | 64GB     | 2Rx8     | TRA564G72D836O        | Hynix M    | 36-51-51-112     | 1.4V    | 6400   |
    | 7200  | V-COLOR     | 64GB     | 2Rx8     | TRAL564G72D836        | Hynix M    | 36-51-51-112     | 1.4V    | 6400   |
    | 7200  | V-COLOR     | 64GB     | 2Rx8     | TRAL564G72D836Q ✅    | Hynix M    | 36-51-51-112     | 1.4V    | 6400   |
    | 7200  | V-COLOR     | 64GB     | 2Rx8     | TRAL564G72D836O       | Hynix M    | 36-51-51-112     | 1.4V    | 6400   |
    | 6400  | Micron      | 64GB     | 2Rx4     | MTC40F2046S1RC64BD2R  | Micron     | CL52             | 1.1V    | 6400   |
    | 6400  | V-COLOR     | 64GB     | 2Rx4     | TR564G64D452          | Hynix A    | 52-52-52-103     | 1.1V    | 6400   |
    | 6400  | V-COLOR     | 64GB     | 2Rx4     | TR564G64D452Q ✅      | Hynix A    | 52-52-52-103     | 1.1V    | 6400   |
    | 6400  | V-COLOR     | 64GB     | 2Rx4     | TR564G64D452O         | Hynix A    | 52-52-52-103     | 1.1V    | 6400   |
    | 6400  | V-COLOR     | 64GB     | 2Rx8     | TRL564G64D852         | Hynix M    | 52-52-52-103     | 1.1V    | 6400   |
    | 6400  | V-COLOR     | 64GB     | 2Rx8     | TRL564G64D852Q        | Hynix M    | 52-52-52-103     | 1.1V    | 6400   |
    | 6400  | V-COLOR     | 64GB     | 2Rx8     | TRL564G64D852O        | Hynix M    | 52-52-52-103     | 1.1V    | 6400   |

    -   [Geizhals.eu](https://geizhals.eu/?cat=ramddr3&xf=7500_DDR5~7569_64GB~7571_6400MT%2Fs~7571_7200MT%2Fs~7761_RDIMM&offset=0&sort=p&promode=true&hloc=at&hloc=de&hloc=pl)
        *All modules: DDR5 RDIMM 288-Pin, JEDEC PC5-51200R, Registered ECC.*
        | Brand      | Model / P/N                  | Capacity | Speed   | CL          | Rank | ECC                  | Price (EUR) |
        |------------|------------------------------|----------|---------|-------------|------|----------------------|-------------|
        | Samsung    | M321R8GA0EB2-CCP             | 64 GB    | 6400 MT/s | CL52      | 2Rx4 | Sideband + On-Die    | 339.00      |
        | Micron     | MTC40F2046S1RC64BR           | 64 GB    | 6400 MT/s | CL52-52-52| 2Rx4 | Sideband + On-Die    | 360.87      |
        | Kingston   | KSM64R52BD4-64MD             | 64 GB    | 6400 MT/s | CL52-52-52| 2Rx4 | Sideband + On-Die    | 417.98      |
        | Micron     | MTC40F2046S1RC64BD2R         | 64 GB    | 6400 MT/s | CL52-52-52| 2Rx4 | Sideband + On-Die    | 440.07      |
        | Kingston   | KSM64R52BD4-64HA             | 64 GB    | 6400 MT/s | CL52-52-52| 2Rx4 | Sideband + On-Die    | 441.00      |
        | G.Skill    | F5-6400R3644E64GQ4-T5N (Kit) | 256 GB   | 6400 MT/s | CL36-44-44-102 | 4x64GB | Sideband + On-Die | 1240.00 |

-   RAM module candidates for the Threadripper 9000 series CPUs, 8 x 32 = 256 GB configuration:
    | Speed  | Supplier       | Capacity | Rank  | Module P/N                | Chip Brand | Timing         | Voltage | Native |
    |--------|----------------|----------|-------|---------------------------|------------|----------------|---------|--------|
    | 8000 ✅ | V-COLOR       | 32GB     | 1Rx8  | TRA532G80S842O            | Hynix M    | 42-60-60-126   | 1.4V    | 6400   |
    | 8000 ✅ | V-COLOR       | 32GB     | 1Rx8  | TRAL532G80S842O           | Hynix M    | 42-60-60-126   | 1.4V    | 6400   |
    | 7200 ✅ | V-COLOR       | 32GB     | 2Rx8  | TRA532G72D834O            | Hynix A    | 34-45-45-96    | 1.35V   | 6400   |
    | 6800   | G.SKILL        | 32GB     | —     | F5-6800R3445G32GE8-ZR5NK  | Hynix      | 34-45-45-108   | 1.4V    | 4800   |
    | 6800   | Kingston FURY  | 32GB     | 2Rx8  | KF568R34RBK8-256          | Hynix A    | 34-44-44-105   | 1.4V    | 4800   |
    | 6800   | Kingston FURY  | 32GB     | 2Rx8  | KF568R34RBK4-128          | Hynix A    | 34-44-44-105   | 1.4V    | 4800   |
    | 6800   | Kingston FURY  | 32GB     | 2Rx8  | KF568R34RB-32             | Hynix A    | 34-44-44-105   | 1.4V    | 4800   |
    | 6400   | Kingston FURY  | 32GB     | 1Rx4  | KF564R32RBK8-256          | Hynix A    | 32-39-39-80    | 1.4V    | 4800   |
    | 6400   | Kingston FURY  | 32GB     | 1Rx4  | KF564R32RBK4-128          | Hynix A    | 32-39-39-80    | 1.4V    | 4800   |
    | 6400   | Kingston FURY  | 32GB     | 1Rx4  | KF564R32RB-32             | Hynix A    | 32-39-39-80    | 1.4V    | 4800   |

-   RAM module candidates for the Threadripper 9000 series CPUs, 4 x 48 = 192 GB configuration:
    | Speed | Supplier   | Capacity | Rank  | Module P/N           | Chip Brand | Timing        | Voltage | Native |
    |-------|------------|----------|-------|----------------------|------------|---------------|---------|--------|
    | 7200  | V-COLOR    | 48GB     | 2Rx8  | TRA548G72D834        | Hynix M    | 34-45-45-96   | 1.35V   | 6400   |
    | 7200  | V-COLOR    | 48GB     | 2Rx8  | TRA548G72D834Q       | Hynix M    | 34-45-45-96   | 1.35V   | 6400   |
    | 7200  | V-COLOR    | 48GB     | 2Rx8  | TRA548G72D834O       | Hynix M    | 34-45-45-96   | 1.35V   | 6400   |
    | 7200  | V-COLOR    | 48GB     | 2Rx8  | TRAL548G72D834       | Hynix M    | 34-45-45-96   | 1.35V   | 6400   |
    | 7200  | V-COLOR    | 48GB     | 2Rx8  | TRAL548G72D834Q      | Hynix M    | 34-45-45-96   | 1.35V   | 6400   |
    | 7200  | V-COLOR    | 48GB     | 2Rx8  | TRAL548G72D834O      | Hynix M    | 34-45-45-96   | 1.35V   | 6400   |
    | 6800  | V-COLOR    | 48GB     | 2Rx8  | TRA548G68D834Q       | Hynix M    | 34-46-46-92   | 1.4V    | 4800   |
    | 6800  | V-COLOR    | 48GB     | 2Rx8  | TRA548G68D834        | Hynix M    | 34-46-46-92   | 1.4V    | 4800   |
    | 6400  | G.SKILL    | 48GB     | —     | F5-6400R3239G48GE8-ZR5NK | Hynix  | 32-39-39-102  | 1.4V    | 4800   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TRA548G64D832Q       | Hynix M    | 32-39-39-102  | 1.4V    | 4800   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TRA548G64D832        | Hynix M    | 32-39-39-102  | 1.4V    | 4800   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TR548G64D852         | Hynix M    | 52-52-52-103  | 1.1V    | 6400   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TR548G64D852Q        | Hynix M    | 52-52-52-103  | 1.1V    | 6400   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TR548G64D852O        | Hynix M    | 52-52-52-103  | 1.1V    | 6400   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TRL548G64D852        | Hynix M    | 52-52-52-103  | 1.1V    | 6400   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TRL548G64D852Q       | Hynix M    | 52-52-52-103  | 1.1V    | 6400   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TRL548G64D852O       | Hynix M    | 52-52-52-103  | 1.1V    | 6400   |

-   RAM module candidates for the Threadripper 7000 series CPUs, 8 x 32 = 256 GB configuration:
    | Speed | Supplier     | Capacity | Rank  | Module P/N                | Chip Brand | Timing           | Voltage | Native |
    |-------|--------------|----------|-------|---------------------------|------------|------------------|---------|--------|
    | 6800  | G.SKILL      | 32GB     | 2Rx8  | F5-6800R3445G32GE8-ZR5NK  | Hynix      | 34-45-45-108     | 1.4V    | 4800   |
    | 6400  | G.SKILL      | 32GB     | 2Rx8  | F5-6400R3239G32GE8-ZR5NK  | Hynix      | 32-39-39-102     | 1.4V    | 4800   |
    | 6400  | Kingston FURY| 32GB     | 2Rx8  | KF564R32RBK8-256          | Hynix A    | —                | 1.4V    | 4800   |
    | 6400  | Kingston FURY| 32GB     | 2Rx8  | KF564R32RBK4-128          | Hynix A    | —                | 1.4V    | 4800   |
    | 6400  | Kingston FURY| 32GB     | 2Rx8  | KF564R32RB-32             | Hynix A    | —                | 1.4V    | 4800   |
    | 6400  | ADATA        | 32GB     | 2Rx8  | AX5R6400C3232G-B          | Hynix A    | 32-39-39-89      | 1.4V    | 5600   |
    | 6400  | ADATA        | 32GB     | 2Rx8  | AX5R6400C3232G-BLAR       | Hynix A    | 32-39-39-89      | 1.4V    | 5600   |
    | 6400  | ADATA        | 32GB     | 2Rx8  | AX5R6400C3232G-SLAR       | Hynix A    | 32-39-39-89      | 1.4V    | 5600   |
    | 6400  | ADATA        | 32GB     | 2Rx8  | AX5R6400C3232G-DTLAR      | Hynix A    | 32-39-39-89      | 1.4V    | 5600   |
    | 6400  | V-COLOR      | 32GB     | 2Rx8  | TRA532G64D832Q            | Hynix A    | 32-39-39-102     | 1.4V    | 4800   |
    | 6400  | V-COLOR      | 32GB     | 2Rx8  | TRA532G64D832             | Hynix A    | 32-39-39-102     | 1.4V    | 4800   |

-   RAM module candidates for the Threadripper 7000 series CPUs, 4 x 48 = 192 GB configuration:
    | Speed | Supplier   | Capacity | Rank  | Module P/N                | Chip Brand | Timing         | Voltage | Native |
    |-------|------------|----------|-------|---------------------------|------------|----------------|---------|--------|
    | 6800  | V-COLOR    | 48GB     | 2Rx8  | TRA548G68D834Q            | Hynix M    | 34-46-46-92    | 1.4V    | 4800   |
    | 6800  | V-COLOR    | 48GB     | 2Rx8  | TRA548G68D834             | Hynix M    | 34-46-46-92    | 1.4V    | 4800   |
    | 6400  | G.SKILL    | 48GB     | 2Rx8  | F5-6400R3239G48GE8-ZR5NK  | Hynix      | 32-39-39-102   | 1.4V    | 4800   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TRA548G64D832Q            | Hynix M    | 32-39-39-102   | 1.4V    | 4800   |
    | 6400  | V-COLOR    | 48GB     | 2Rx8  | TRA548G64D832             | Hynix M    | 32-39-39-102   | 1.4V    | 4800   |

</details>

<details><summary>Motherboard</summary>

-   [Why is the Gigabyte TRX50 AI Top the Most Returned Threadripper Motherboard?](https://www.youtube.com/watch?v=qVeyYnD0BZI)
-   [BIGGEST AI Motherboard EVER Created & why it's THE BEST!! feat. Gigabyte TRX50 AI TOP](https://www.youtube.com/watch?v=u3Fe_3qhpCU)

</details>

## Desktop PC

<details><summary>Details</summary>

-   10 GB LLM model expected CPU-only performance: Prompt processing: 125 token/s, Token generation: 8 token/s
-   Supports 1 PCIe x16 GPU, or 2 PCIe x8 GPUs.

| Component   | Model                                              | Price each<br>[HUF] | Price subtotal<br>[HUF] |
|-------------|----------------------------------------------------|--------------------:|------------------------:|
| CPU         | AMD Ryzen 9 9950X3D                                |       1,120,491     |           1,120,491     |
| RAM         |     |          76,020     |             912,240     |
| SSD         | Samsung PM9A3 1.9TB NVMe PCIe Gen4 V6 M.2 22x110   |         129,910     |             129,910     |
| Motherboard |                            |         328,912     |             328,912     |
| CPU cooler  | Noctua NH-D15 G2                                   |               |              23,990     |
| PSU         | Seasonic Prime PX-2200 2200W 80 PLUS Platinum      |         212,990     |             212,990     |
| Chassis     | Fractal Design Torrent                             |          75,600     |              75,600     |
| **Total**   |                                                    |                     |         **2,804,133**   |


### AMD Ryzen 9 9950X3D

#### Specs

-   CPU: AMD Ryzen 9 9950X3D
    ([Wikipedia](https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Granite_Ridge_(9000_series,_Zen_5_based)))
    ([AMD](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html))
    ([TechPowerUp](https://www.techpowerup.com/cpu-specs/ryzen-9-9950x3d.c3993))
    -   Max. Memory: 192 GB
    -   Max Memory Speed
        -   2x1R    DDR5-5600
        -   2x2R    DDR5-5600
        -   4x1R    DDR5-3600
        -   4x2R    DDR5-3600
    -   Max memory bandwidth:
        -   2 modules -> 5600 MT/s -> 2 x 8 x 5.6 = 89.6 GB/s
        -   4 modules -> 3600 MT/s -> 2 x 8 x 3.6 = 57.6 GB/s
            -   ❗**Note:** This is not sufficient to saturate the PCIe 5.0 x16 lanes, which require 63 GB/s.
    -   128MB L3 cache
        -   192MB L3 cache variant is [available](https://www.techpowerup.com/review/future-hardware-releases/#ryzenx3ddualcc)
            ([soon](https://www.techpowerup.com/339579/amd-readies-16-core-ryzen-9000x3d-cpu-with-192-mb-l3-cache-and-200-w-tdp))
        -   Default TDP: 170W
-   CPU cooler
    -   [Noctua NH-D15 G2](https://noctua.at/en/nh-d15-g2/specification)
        -   [CPU compatibility](https://ncc.noctua.at/cpus/model/AMD-Ryzen-9-9950X3D-1864): OK
        -   Height (with fan): 168 mm
            -   OK with [Fractal Design Define 7](https://www.fractal-design.com/products/cases/define/define-7/) case
        -   RAM clearance in dual fan mode:
            -   32mm with 140mm fan [168mm total height]
            -   52mm with 120mm fan [168mm total height]
-   Chipsets
    ([AMD doc](https://www.amd.com/en/products/processors/chipsets/am5.html#specs))
    ([TechPowerUp](https://www.techpowerup.com/cpu-specs/ryzen-9-9950x3d.c3993#gallery-5)):
    ([Wikipedia](https://en.wikipedia.org/wiki/List_of_AMD_chipsets#AM5_chipsets))
    -   Has 24 PCIe 5.0 lanes: X870E, X870, X670E, B650E
    -   Has USB 4.0: X870E, X870
    -   X870E:
        -   USB 4.0
        -   2 SUPERSPEED USB 20Gbps
        -   PCIe 5.0 1x16 or 2x8
        -   1x4 PCIe 5.0 plus 4x PCIe GPP
-   Motherboard:
    -   [ASRock X870E Taichi](https://www.asrock.com/mb/AMD/X870E%20Taichi/#Specification)
        -   [Tom's Hardware review](https://www.tomshardware.com/pc-components/motherboards/asrock-x870e-taichi-review)
        -   E-ATX
        -   2 x PCIe 5.0 x16 Slots (PCIE1 and PCIE2), support x16 or x8/x8 modes
        -   Supports DDR5 ECC/non-ECC, un-buffered memory up to 8200+(OC)*
        -   Max. capacity of system memory: 256GB
        -   Supports RAID 0, RAID 1 and RAID 10 for M.2 NVMe storage devices
        -   2 x 8 pin 12V Power Connectors (Hi-Density Power Connector)
        -   Dual RTX 5090 readiness:
            -   Supported electrically (PCIe 5.0 x8/x8 for two GPUs).
            -   No supplemental PCIe slot power header is advertised; ensure robust PSU and airflow.
        -   ❗**Note:** May burn the CPU with unfortunate PBO settings
    -   [ASUS ROG Crosshair X870E Hero](https://rog.asus.com/motherboards/rog-crosshair/rog-crosshair-x870e-hero-model/)
        -   Dual RTX 5090 readiness:
            -   Supported electrically (x8/x8).
            -   Auxiliary PCIe slot power header may vary by revision—verify
                in the manual; spacing/airflow are critical.
    -   [ASUS ROG Crosshair X870E Extreme](https://rog.asus.com/us/motherboards/rog-crosshair/rog-crosshair-x870e-extreme/)
        -   Dual RTX 5090 readiness:
            -   High confidence. Flagship boards typically include an
                auxiliary PCIe slot power header—verify in the manual; strong
                pick for dual GPUs.
    -   [Gigabyte X870E Aorus Xtreme AI TOP](https://www.gigabyte.com/Motherboard/X870E-AORUS-XTREME-AI-TOP-rev-1x)
        -   Dual RTX 5090 readiness:
            -   High confidence. AORUS XTREME-class boards usually include an
                auxiliary PCIe slot power header—verify in the manual; strong
                pick for dual GPUs.
    -   [MSI MEG X870E GODLIKE](https://www.msi.com/Motherboard/MEG-X870E-GODLIKE)
        -   [Datasheet](https://storage-asset.msi.com/datasheet/mb/global/MEG-X870E-GODLIKE.pdf)
        -   E-ATX
        -   4x DDR5 UDIMM, Maximum Memory Capacity 256GB
        -   Memory Support DDR5 9000 - 5600 (OC) MT/s / 5600 - 4800 (JEDEC) MT/s
        -   Ryzen™ 9000 Series Processors max. overclocking frequency:
            -   1DPC 1R Max speed up to 8400+ MT/s
            -   1DPC 2R Max speed up to 6400+ MT/s
            -   2DPC 1R Max speed up to 6400+ MT/s
            -   2DPC 2R Max speed up to 6400+ MT/s
        -   Supports AMD POR Speed and JEDEC Speed
        -   Supports Memory Overclocking and AMD EXPO
        -   Supports Dual-Channel mode
        -   Supports Non-ECC, Un-buffered memory
        -   Supports CUDIMM, Clock Driver bypass mode only*
        -   Dual RTX 5090 readiness:
            -   Yes. Includes a supplemental PCIe slot power header ("PCIe
                Supplemental Power"); best-in-class choice for dual high-power
                GPUs.
        -   RAM modules supported by the MSI MEG X870E GODLIKE motherboard: https://www.msi.com/Motherboard/MEG-X870E-GODLIKE/support#mem
            | Vendor      | Model                              | SPD Speed (MHz) | Supported Speed (MHz) | Voltage (V) | Sided | Size (GB) | 1/2/4 DIMM |
            |-------------|------------------------------------|-----------------|-----------------------|-------------|-------|-----------|------------|
            | G.SKILL     | F5-6000J3644D64GX4-TR5NS           | 5600            | 6000                  | 1.25        | DUAL  | 64        | √ √ √      |
            | G.SKILL     | F5-6000J3644D64GX4-TZ5NR           | 5600            | 6000                  | 1.25        | DUAL  | 64        | √ √ √      |
            | G.SKILL     | F5-6000J3644D64GX4-FX5             | 5600            | 6000                  | 1.25        | DUAL  | 64        | √ √ √      |
            | G.SKILL     | F5-6000J3444F64GX4-FX5             | 5600            | 6000                  | 1.35        | DUAL  | 64        | √ √ √      |
            | Kingston    | KF556C40BBK2-128                   | 4800            | 5600                  | 1.25        | DUAL  | 64        | √ √ √      |
            | Kingston    | KF556C36BBEAK2-128                 | 4800            | 5600                  | 1.25        | DUAL  | 64        | √ √ √      |
            | Kingston    | KF556C36BBE-64                     | 4800            | 5600                  | 1.25        | DUAL  | 64        | √ √ √      |
            | Kingston    | KF556C36BBEA-64                    | 4800            | 5600                  | 1.25        | DUAL  | 64        | √ √ √      |
            | Kingston    | KF556C36BBEK2-128                  | 4800            | 5600                  | 1.25        | DUAL  | 64        | √ √ √      |
            | Kingston    | KF556C40BBA-64                     | 4800            | 5600                  | 1.25        | DUAL  | 64        | √ √ √      |
            | Crucial     | CT64G56C46U5.M16B1                 | 5600            | 3600                  | 1.1         | DUAL  | 64        | √ √ √      |
            | Crucial     | CP64G56C46U5.M16B1                 | 5600            | 3600                  | 1.1         | DUAL  | 64        | √ √ √      |
            | BIWIN       | OCBXL59264DW1-Q30FB                | 5600            | **6400**              | 1.4         | DUAL  | 48        | √ √ √      |
            | ADATA(XPG)  | AX5U6000C2848G-BB300X4             | 4800            | 6000                  | 1.4         | DUAL  | 48        | √ √ √      |
            | ADATA(XPG)  | AX5U6000C2848G-BW300X4             | 4800            | 6000                  | 1.4         | DUAL  | 48        | √ √ √      |
            | KingBank    | KPR548G60C28-P                     | 4800            | 6000                  | 1.4         | DUAL  | 48        | √ √ √      |
            | ADATA(XPG)  | AX5U6000C2848G-BLABBKX4            | 4800            | 6000                  | 1.4         | DUAL  | 48        | √ √ √      |
            | BIWIN       | OCBXL59260DW1-Q28FB                | 5600            | 6000                  | 1.4         | DUAL  | 48        | √ √ √      |
            | CORSAIR     | CMH192GX5M4B5200C38 ver5.53.13     | 4800            | 5200                  | 1.25        | DUAL  | 48        | √ √ √      |
            | CORSAIR     | CMK192GX5M4B5200C38 ver3.53.02     | 4800            | 5200                  | 1.25        | DUAL  | 48        | √ √ √      |
            | KLEVV       | KD5LGUD80-56G4600                  | 5600            | 3600                  | 1.1         | DUAL  | 48        | √ √ √      |
            | Crucial     | CP48G56C46U5.C16B                  | 5600            | 3600                  | 1.1         | DUAL  | 48        | √ √ √      |
            | Kingston    | KVR56U46BD8-48                     | 5600            | 3600                  | 1.1         | DUAL  | 48        | √ √ √      |
-   RAM:\
   \ -   CPU max RAM: [192 GB](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html)
    -   Motherboard: 2x/4x DDR5 DIMM slots
    -   [128 GB](https://www.arukereso.hu/memoria-modul-c3577/f:kapacitas-128-gb,tipus-pc-memoria/?orderby=1): 360 EUR
        -   RAM BW upper limit: 2 x 8 x 5.6 = 89.6 GB/s
    -   [192 GB](https://www.arukereso.hu/memoria-modul-c3577/f:kapacitas-192-gb,tipus-pc-memoria/): 760 EUR
        -   [RAM BW upper limit](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html):
            2 channels x 8 x 3.6 MT/s = 57.6 GB/s
            -   ❗**Note:** 192 GB (vs 128 GB) system RAM will **reduce** 10 GB
                LLM model CPU-only token generation speed from 8 token/s down
                to 5 token/s maximum. Overclocking may improve a few token/s,
                but no substantial improvement.
            -   ❗**Note:** This practically limits the system RAM to 128 GB.
-   SSD:
    -   [Samsung 990 PRO 4TB (MZ-V9P4T0BW)](https://www.techpowerup.com/ssd-specs/samsung-990-pro-4-tb.d863)
-   PSU:
    -   CPU + RAM + SSD + motherboard: ~300 W
    -   GPU:
        -   NVIDIA GeForce RTX 5090: 575 W
    -   Total for a 2-GPU setup:
        -   300 + 2 x 575 = 1450 W
        -   Headroom: 50%
        -   Required power supply: 2175 W
    -   [Seasonic Prime PX-2200 2200W 80 PLUS Platinum](https://seasonic.com/atx3-prime-px-2200/)
        -   Total continuous power 	2200 W
-   Case: E-ATX
    -   [Fractal Design Define 7](https://www.fractal-design.com/products/cases/define/define-7/)
        -   Total fan mounts: 9 x 120/140 mm
        -   Front fan: 3 x 120/140 mm (2 x Dynamic X2 GP-14 included)
        -   Top fan: 3 x 120/140 mm
        -   Rear fan: 1 x 120/140 mm (1 x Dynamic X2 GP-14 included)
        -   Bottom fan: 2 x 120/140 mm
        -   GPU max length:
            -   Storage layout: 290 mm
            -   Open layout: 470 mm (445 mm w/ front fan)
        -   CPU cooler max height: 185 mm
            -   OK with [Noctua NH-D15 G2](https://noctua.at/en/nh-d15-g2/specification) CPU cooler
        -   Front radiator: Up to 360/280 mm
        -   Top radiator: Up to 360/420 mm
        -   Rear radiator: 120 mm
        -   Vertical GPU Support (with Flex B-20 or Flex VRC-25): 65mm total
            clearance, standard 2-slot GPU (<38mm thickness) recommended for
            optimum cooling
        -   Case dimensions (LxWxH): 547 x 240 x 475 mm
-   GPU:
    -   AM5/X870E platforms split the CPU PEG lanes to x8/x8 for dual GPUs; x16/x16 isn’t available.
    -   Physical fit: most RTX 5090 cards are 3–3.5-slot wide. Fitting two often requires:
        -   A case with 8+ expansion slots and generous bottom clearance
        -   Motherboard slot spacing that leaves at least 3 full slots between
            x16_1 and x16_2
    -   Power: plan for a high-end PSU (typically 1600–2000W) and adequate
        12V‑2×6 connectors; some boards (e.g., MSI GODLIKE) include a
        supplemental PCIe slot power header that helps stability with dual
        GPUs.
    -   No SLI/NVLink for 5090; dual-GPU is for compute (CUDA/ML), not gaming AFR.

#### Price

-   CPU:
    -   [AMD Ryzen 9 9950X3D](https://www.arukereso.hu/processzor-c3139/?orderby=1&st=9950x3d): 700 EUR
-   CPU cooler:
    -   [Noctua NH-D15 G2](https://www.arukereso.hu/szamitogep-huto-c3094/noctua/nh-d15-g2-p1101386203/): 180 EUR
-   RAM:
    -   [128 GB](https://www.arukereso.hu/memoria-modul-c3577/f:kapacitas-128-gb,tipus-pc-memoria/?orderby=1): 360 EUR
-   SSD:
    -   [Samsung 990 PRO 4TB (MZ-V9P4T0BW)](https://belso-ssd-meghajto.arukereso.hu/samsung/990-pro-4tb-mz-v9p4t0bw-p1002242350/): 300 EUR
-   Motherboard:
    -   [ASRock X870E Taichi](https://www.arukereso.hu/alaplap-c3128/?st=ASRock+X870E+Taichi): 490 EUR
    -   [Ausus ROG Crosshair X870E Hero](https://www.arukereso.hu/alaplap-c3128/?st=ROG+Crosshair+X870E): 650 EUR
    -   [Ausus ROG Crosshair X870E Extreme](https://www.arukereso.hu/alaplap-c3128/?st=ROG+Crosshair+X870E): 1300 EUR
    -   [Gigabyte X870E Aorus Xtreme AI TOP](https://www.arukereso.hu/alaplap-c3128/gigabyte/x870e-aorus-xtreme-ai-top-p1163326837/): 800 EUR
    -   [MSI Meg X870E Godlike](https://www.arukereso.hu/alaplap-c3128/msi/meg-x870e-godlike-p1160493943/): 1300 EUR
-   PSU
    -   [Seasonic Prime PX-2200 2200W 80 PLUS Platinum](https://www.arukereso.hu/tapegyseg-c3158/seasonic/prime-px-2200-2200w-80-plus-platinum-p1129871905/): 550 EUR
-   Case
    -   [Fractal Design Define 7](https://www.arukereso.hu/szamitogep-haz-c3085/fractal-design/define-7-black-fd-c-def7a-01-p545013072/#): 240 EUR
-   Total price without GPU:
    -   700 + 180 + 360 + 300 + 490 + 550 + 240 = 2820 EUR minimum.
    -   2520 + 300 ~= 3100 EUR with Gigabyte X870E Aorus Xtreme AI TOP motherboard.



Desktop CPU Links:
-   https://www.anandtech.com/show/21524/the-amd-ryzen-9-9950x-and-ryzen-9-9900x-review/10
-   https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html
-   https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html
-   https://www.amd.com/en/products/processors/chipsets/am5.html
-   https://skatterbencher.com/2025/03/11/skatterbencher-85-ryzen-9-9950x3d-overclocked-to-5900-mhz/

</details>
<div class="page"/>

## Embedded / mobile CPU

<details><summary>Details</summary>

Technical information:
-   https://www.techpowerup.com/cpu-specs/ryzen-ai-max-pro-395.c3998
-   https://www.amd.com/en/products/processors/laptop/ryzen-pro/ai-max-pro-300-series/amd-ryzen-ai-max-plus-pro-395.html
-   https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html
    >   128GB of unified memory – out of which up to 96GB can be converted to
    >   VRAM through AMD Variable Graphics Memory
-   https://www.tomshardware.com/pc-components/cpus/more-affordable-strix-halo-model-emerges-early-ryzen-ai-max-385-geekbench-result-reveals-an-eight-core-option

Discussion about adding an eGPU to the main board:
-   https://www.reddit.com/r/LocalLLaMA/comments/1kedbv7/ryzen_ai_max_395_a_gpu/
-   Note that the Framework mainboard has [only 1 x PCIe x4 slot](https://frame.work/hu/en/products/framework-desktop-mainboard-amd-ryzen-ai-max-300-series?v=FRAFMK0006)

Expected performance for a 10 GB LLM model:
-   Prompt processing: 84 token/s
-   Token generation: 11 token/s (100-200 GB/s RAM throughput)

### GMKtec EVO-X2 AMD Ryzen™ AI Max+ 395 Mini PC - 2000 EUR

-   [Spec](https://de.gmktec.com/en/products/gmktec-evo-x2-amd-ryzen%E2%84%A2-ai-max-395-mini-pc-1?variant=51610049380536#tab-spezifikation):
    -   AMD Ryzen™ AI Max+ 395
    -   Radeon 8060S graphics
    -   128 GB RAM
-   [Price](https://de.gmktec.com/en/products/gmktec-evo-x2-amd-ryzen%E2%84%A2-ai-max-395-mini-pc-1?variant=51610049380536): 2000 EUR

### Framework Desktop mini PC - 3200 EUR

-   [Spec](https://frame.work/hu/en/desktop?slug=desktop-diy-amd-aimax300&tab=specs):
    -   AMD Ryzen™ AI Max+ 395 (soldered)
    -   Noctua CPU Fan - NF-A12x25 HS-PWM
    -   Form Factor: FlexATX
    -   Wattage: 400W
    -   128GB LPDDR5x-8000 memory (soldered)
    -   Memory Bus width: 256-bit
    -   Memory Speed: 8000 MT/s
    -
-   [Price](https://frame.work/hu/en/products/desktop-diy-amd-aimax300/configuration/new):
    -   3200 EUR
    -   Ships 2025 Q4

### Framework Desktop mainboard only - 2100 EUR

-   [Spec](https://frame.work/hu/en/products/framework-desktop-mainboard-amd-ryzen-ai-max-300-series?v=FRAFMK0006):
    -   AMD Ryzen™ AI Max+ 395 (soldered)
    -   128GB LPDDR5x-8000 memory (soldered)
    -   Memory Bus width: 256-bit
    -   Memory Speed: 8000 MT/s
    -   Mini-ITX case,
    -   Power supply: 500W or higher ATX, SFX, or FlexATX
    -   Fan: 120mm, with a minimum of 85.1 CFM air flow and 3.82 mmH2O air pressure
    -   Can be used in a cluster:
        >   Run legit, state-of-the-art AI models like Llama 70B right on your
        >   desk with up to 96GB of graphics addressable memory and a 256-bit
        >   memory bus.
        >
        >   Framework Desktop has 5Gbit Ethernet along with two USB4 ports,
        >   allowing networking multiple together to run even larger models with
        >   llama.cpp RPC. Grab a few Mainboards and build it into your own
        >   mini-racks or standard rackmount server cases for high density.
    -   PCIe:
        >   1 x PCIe x4 slot (not exposed on default case)
-   [Price](https://frame.work/hu/en/products/framework-desktop-mainboard-amd-ryzen-ai-max-300-series?v=FRAFMK0006):
    -   2100 EUR

### HP ZBook Ultra G1a Mobile Workstation PC - 3600 EUR

-   [Spec](https://www.hp.com/us-en/shop/pdp/hp-zbook-ultra-14-inch-g1a-mobile-workstation-pc-wolf-pro-security-edition-p-bn2v3ua-aba-1#techSpecs):
    -   AMD Ryzen™ AI Max+ PRO 395 (up to 5.1 GHz max boost clock, 64 MB L3 cache, 16 cores, 32 threads)[6,7]
    -   128 GB LPDDR5x-8533 MT/s (onboard)
    -   Graphics: Integrated: AMD Radeon™ 8060S Graphics
    -   SSD: 2 TB PCIe Gen4 NVMe™ TLC SSD
    -   Display: 14" diagonal, 2.8K (2880 x 1800), OLED, touch, IPS, BrightView, Low Blue Light, 400 nits, 100% DCI-P3
-   [Price](https://www.hp.com/us-en/shop/pdp/hp-zbook-ultra-14-inch-g1a-mobile-workstation-pc-wolf-pro-security-edition-p-bn2v3ua-aba-1):
    -   4200 USD ~= 3600 EUR

### Other embedded/mobile options

Some other links:
-   https://store.minisforum.com/products/elitemini-ai370
-   https://minisforumpc.eu/en/products/ai-x1-pro-mini-pc?variant=51875206496622
-   https://www.hp.com/us-en/workstations/z2-mini-a.html
-   https://www.techpowerup.com/333983/sapphire-develops-edge-ai-mini-pc-series-with-amd-ryzen-ai-300-targeting-gamers-and-creatives
-   https://www.reddit.com/r/LocalLLaMA/comments/1judxsq/gmktec_evox2_powered_by_ryzen_ai_max_395_to/

</details>
<div class="page"/>

## GPUs

<details><summary>Details</summary>

### NVIDIA RTX PRO 6000 Blackwell (96GB)

Specs ([TechPowerUp](https://www.techpowerup.com/gpu-specs/?q=%226000+Blackwell%22)):
-   Memory Size: 96 GB
-   Memory Bus: 512 bit
-   Bandwidth: 1.79 TB/s
-   Shading Units: 24064
-   Tensor Cores: 752
-   L1 Cache: 128 KB (per SM)
-   L2 Cache: 128 MB
-   FP16 (half): 126.0 TFLOPS (1:1)
-   FP32 (float): 126.0 TFLOPS
-   Bus Interface: PCIe 5.0 x16
    -   63 GB/s required PCIe bandwidth
-   Variants:
    -   [NVIDIA RTX PRO 6000 Blackwell](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4272)
        -   TDP: 600 W
        -   Suggested PSU: 1000 W
        -   Open-air fan
    -   [NVIDIA RTX PRO 6000 Blackwell Server](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell-server.c4274)
        -   TDP: 600 W
        -   Suggested PSU: 1000 W
        -   Fan: Bowler type
    -   [NVIDIA RTX PRO 6000 Blackwell Max-Q](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell-max-q.c4273)
        -   TDP: 300 W
        -   Suggested PSU: 700 W
        -   Fan: Bowler type

### NVIDIA GeForce RTX 5090 (32GB)

-   [Power](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216)
    >   Being a dual-slot card, the NVIDIA GeForce RTX 5090 draws power from
    >   1x 16-pin power connector, with power draw rated at 575 W maximum.

### NVIDIA GeForce RTX 5070 Ti SUPER (24GB)

[TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5070-ti-super.c4312)

### NVIDIA GeForce RTX 5080 SUPER (24GB)

[TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5080-super.c4302)

### AMD Radeon RX 7900 XTX - 1000 EUR

Specs ([TechPowerUp](https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941)):
-   Memory Size: 24 GB
-   Memory Bus: 384 bit
-   Bandwidth: 960.0 GB/s
-   Shading Units: 6144
-   Compute Units: 96
-   FP16 (half): 122.8 TFLOPS (2:1)
-   FP32 (float): 61.39 TFLOPS
-   FP64 (double): 1.918 TFLOPS (1:32)
-   TDP: 355 W
-   Suggested PSU: 750 W

Price:
-   [Árukereső](https://www.arukereso.hu/videokartya-c3142/f:amd-radeon-video-chipset,rx-7900-xtx/?orderby=1)
    -   341,000 to 419,000 HUF (860 to 1100 EUR)

### AMD Radeon™AI PRO R9700

Specs
([TechPowerUp](https://www.techpowerup.com/gpu-specs/radeon-ai-pro-r9700.c4290))
([AMD](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html))
-   GPU Architecture: AMD RDNA™ 4
-   Memory Size:    32 GB
-   Memory Type:    GDDR6
-   Memory Bus:    256 bit
-   Bandwidth:    644.6 GB/s
-   Shading Units: 4096
-   Compute Units:    64
-   Matrix Cores:     128
-   TDP:    300 W
-   Suggested PSU: 700 W
-   Peak Single Precision (FP32 Vector) Performance:    47.8 TFLOPs
-   Peak Half Precision (FP16 Vector) Performance:    95.7 TFLOPs
-   Peak Half Precision (FP16 Matrix) Performance:    191 TFLOPs
-   Peak Half Precision (FP16 Matrix) Performance with Structured Sparsity:     383 TFLOPs
-   Peak 8-bit Precision (FP8 Matrix) Performance (E5M2, E4M3):     383 TFLOPs
-   Peak 8-bit Precision (FP8 Matrix) Performance with Structured Sparsity (E5M2, E4M3):     766 TFLOPs
-   Peak 8-bit Precision (INT8 Matrix) Performance:     383 TOPs
-   Peak 8-bit Precision (INT8 Matrix) Performance with Structured Sparsity:    766 TOPs
-   Peak 4-bit Precision (INT4 Matrix) Performance:     766 TOPs
-   Peak 4-bit Precision (INT4 Matrix) Performance with Structured Sparsity:     1531 TOPs

Variants:
-   [ASUS Turbo Radeon AI PRO R9700](https://www.techpowerup.com/gpu-specs/asus-turbo-radeon-ai-pro-r9700.b12819)
    -   [TURBO-AI-PRO-R9700-32G](https://www.asus.com/uk/motherboards-components/graphics-cards/turbo/turbo-ai-pro-r9700-32g/)
-   [GIGABYTE Radeon AI PRO R9700 AI TOP](https://www.techpowerup.com/gpu-specs/gigabyte-radeon-ai-pro-r9700-ai-top.b12619)
    -   [GV-R9700AI TOP-32GD](https://www.gigabyte.com/Graphics-Card/GV-R9700AI-TOP-32GD)
-   [PowerColor Radeon AI PRO R9700](https://www.techpowerup.com/gpu-specs/powercolor-radeon-ai-pro-r9700.b12666)
    -   [AI PRO R9700 32G](https://www.powercolor.com/product-detail258.htm)
-   [Sapphire AMD RADEON AI PRO R9700 32GB](https://www.sapphiretech.com/en/commercial/radeon-ai-pro-r9700)

### eGPU Dock

#### Peladn Link S-3

[Specs](https://peladn.com/products/graphics-card-docking-station-1)
-   I/O:    Thunderbolt3*2[(1),75W (2),25W]
-   Power:  ATX/SFX
-   Size:   300*220*180mm
-   Support:    Basically meet most existing models of graphics cards(It is recommended that AMD graphics card use RX570 or above, and NVIDIA graphics card use GTX 1060 or above)
-   System Requirement: Windows 10 64 Bit(Support AMD&NVIDIA graphics cards) MacOS > 10.13.4(Only AMD graphics cards are supported)
-   Computer Requirement:   Thunderbolt 3 port and support external GPU
-   Transfer Speed: 40 Gbps

</details>

