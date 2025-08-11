# Hardware alternatives for running machine learning models locally

-   [Embedded / mobile CPU](#embedded--mobile-cpu)
-   [Desktop PC](#desktop-pc)
-   [Workstation](#workstation)
-   [Server](#server)

The following GPU alternatives are listed:
-   NVIDIA RTX PRO 6000 Blackwell (96GB)
-   NVIDIA GeForce RTX 5090 (32GB)
-   NVIDIA GeForce RTX 5070 Ti SUPER (24GB)
-   NVIDIA GeForce RTX 5080 SUPER (24GB)
-   AMD Radeon RX 7900 XTX (24GB)
-   AMD Radeon AI PRO R9700

PLease refer to the '[Which GPU(s) to Get for Deep Learning: My Experience
and Advice for Using GPUs in Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)'
blog post by Tim Dettmers.
>   Tensor Cores are most important, followed by memory bandwidth of a GPU,
>   the cache hierarchy, and only then FLOPS of a GPU

See also the [Build your own machine](https://huggingface.co/docs/transformers/perf_hardware) guide on HuggingFace.

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

## Desktop PC

<details><summary>Details</summary>

-   10 GB LLM model expected CPU-only performance: Prompt processing: 125 token/s, Token generation: 8 token/s
-   Supports 1 PCIe x16 GPU, or 2 PCIe x8 GPUs.

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
    ([Image](assets/amd-am5.png))
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
        -   Dual RTX 5090 readiness:
            -   Yes. Includes a supplemental PCIe slot power header ("PCIe
                Supplemental Power"); best-in-class choice for dual high-power
                GPUs.
-   RAM:
    -   CPU max RAM: [192 GB](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html)
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

## Workstation

## Server

<details><summary>Requirements</summary>

-   The target system must support at least 2 NVIDIA RTX PRO 6000 Blackwell (96GB) GPUs
    -   Required system RAM: 1-2 x total GPU VRAM
        -   2x 96 GB GPUs: 192 GB minimum, 384 GB ideally
        -   4x 96 GB GPUs: 384 GB minimum, 768 GB ideally
        -   6x 96 GB GPUs: 576 GB minimum, 1152 GB ideally
    -   Memory modules:
        -   Note that from the Genoa (AMD EPYC 4004, 8004, 9004) platform on,
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
    -   PSU:
        -   2 GPUs: min. 2000 W
        -   4 GPUs: min. 4000 W
        -   6 GPUs: min. 6000 W
        -   plus the system requirement (motherboard + CPU + cooler + RAM + SSD)
    -   CPU: AMD EPYC 9004 / 9005 (SP5 socket)
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
            -   [Zen 4](https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena)):
                Any AMD EPYC 9004 CPU, except 9124, 9224, 9254, 9334, because n_CCD < 8
            -   [Zen 5](https://en.wikipedia.org/wiki/Epyc#Fifth_generation_Epyc_(Grado,_Turin_and_Turin_Dense)):
                Any AMD EPYC 9005 CPU, except 9015, 9115, 9135, 9255, 9335, 9365, because n_CCD < 8
            -   [EPYC 9004 Series CPU Positioning](https://hothardware.com/Image/Resize/?width=1170&height=1170&imageFile=/contentimages/Article/3257/content/big_epyc-cpu-positioning.jpg)
                ![EPYC 9004 Series CPU Positioning.png](<assets/EPYC 9004 Series CPU Positioning.png>)
            -   Complete list of CPU candidates:
                -   9354, 9354P, 9174F, 9184X, 9274F, 9374F, 9384X, 9474F,
                    9454, 9454P, 9534, 9554, 9554P, 9634, 9654, 9654P, 9684X,
                    9734, 9754S, 9754, 9645, 9745, 9825, 9845, 9965, 9175F,
                    9275F, 9355P, 9355, 9375F, 9455P, 9455, 9475F, 9535,
                    9555P, 9555, 9575F, 9565, 9655P, 9655, 9755, 4245P, 4345P,
                    4465P, 4545P, 4565P, 4585PX
            -   If CPU inference is not a priority, then lower core count and
                thus lower DTP/cDTP is sufficient.
            -   All CPUs below the 240 TDP line have less than 8 CCDs, so they cannot utilize the available RAM bandwidth.
            -   CPU candidates whose configurable TDP (cTDP) is
                [in the 240-300 W range](https://www.amd.com/en/products/specifications/server-processor.html):
                -   Zen 4: 9354, 9354P, 9454, 9454P, 9534, 9634
                -   Zen 5: 9355P, 9355, 9365, 9455P, 9455, 9535

</details> <!-- Requirements -->

<details><summary>Specs</summary>

-   CPU: AMD EPYC 9354
    ([Wikipedia](https://en.wikipedia.org/wiki/Epyc#Fourth_generation_Epyc_(Genoa,_Bergamo_and_Siena))))
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
-   CPU cooler
    -   [Noctua NH-D15 G2](https://noctua.at/en/nh-d15-g2/specification)
        -   [CPU compatibility](https://ncc.noctua.at/cpus/model/AMD-Ryzen-9-9950X3D-1864): OK
        -   Height (with fan): 168 mm
            -   OK with [Fractal Design Define 7](https://www.fractal-design.com/products/cases/define/define-7/) case
        -   RAM clearance in dual fan mode:
            -   32mm with 140mm fan [168mm total height]
            -   52mm with 120mm fan [168mm total height]
-   Motherboard:
    -   Supermicro
        [H13SSL‑NT](https://www.supermicro.com/en/products/motherboard/H13SSL-NT) /
        [H13SSL‑N](https://www.supermicro.com/en/products/motherboard/h13ssl-n):
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
    -   [ASRock Rack GENOAD8QM3‑2T/BCM](https://www.asrockrack.com/general/productdetail.asp?Model=GENOAD8QM3-2T/BCM#Specifications) /
        -   Form factor: CEB
        -   dual 10 GbE,
        -   MCIO NVMe support,
        -   PCIe 5.0 expansion with clean board layout suitable for GPUs.
        -   Popular in the community as a stable SP5 choice ([Newegg.com][3.7]).
    -   [ASRock Rack GENOAD8UD‑2T/X550](https://www.asrockrack.com/general/productdetail.asp?Model=GENOAD8UD-2T/X550#Specifications):
        -   Dimensions	10.4" x 10.5"
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
    -   [GIGABYTE MZ73-LM2](https://www.gigabyte.com/us/Enterprise/Server-Motherboard/MZ73-LM2-rev-3x)
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
-   RAM:
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

[3.7]: https://www.newegg.com/p/pl?N=100007629+601411369&srsltid=AfmBOorDHas0epelrMNy2kXSwLPh7xgASlrIeIDU2XXqA3ZYBGRT9cm3&utm_source=chatgpt.com "Socket SP5 Server Motherboards | Newegg.com"
[3.12]: https://www.newegg.com/p/pl?N=100007629+601411369&utm_source=chatgpt.com "Socket SP5 Server Motherboards | Newegg.com"
[3.13]: https://www.reddit.com/r/homelab/comments/1h1iprj?utm_source=chatgpt.com "Epyc 97x4 Genoa Motherboard"

</details> <!-- Specs -->

<details><summary>Prices</summary>

AMD EPYC Zen 4 / Zen 5 processors with a TDP of less than 300 W, as of 2025.08.09

| CPU                |  Price [HUF] |  Price [EUR] |
|--------------------|-------------:|-------------:|
| **Zen 4**          |              |              |
| [9354][pr9354]     |      601,460 |         1500 |
| [9354P][pr9354P]   |      750,362 |         1900 |
| [9454][pr9454]     |      928,075 |         2300 |
| [9454P][pr9454P]   |      829,174 |         2100 |
| [9534][pr9534]     |      753,990 |         1900 |
| [9634][pr9634]     |    1,372,177 |         3400 |
| **Zen 5**          |              |              |
| [9355][pr9355]     |    1,259,666 |         3200 |
| 9355P              |              |              |
| 9365               |              |              |
| 9455P              |              |              |
| 9455               |              |              |
| 9535               |    2,300,852 |         5800 |

[pr9354]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9354-32-core-3-25ghz-sp5-tray-100-000000798-p923300802/
[pr9354P]: https://kontaktor.hu/amd_epyc_9354p_processor_325_ghz_256_mb_l3_380250
[pr9454]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9454-48-core-2-75ghz-sp5-tray-100-000000478-p923301999/
[pr9454P]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9454p-48-core-2-75ghz-sp5-tray-100-000000873-p981692184/
[pr9534]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9534-2-45ghz-sp5-tray-100-000000799-p985734456/
[pr9634]: https://smicro.hu/amd-epyc-genoa-9634-dp-up-84c-168t-2-25g-384mb-290w-sp5-100-000000797-5
[pr9355]: https://www.arukereso.hu/processzor-c3139/amd/epyc-9355-32-core-3-55ghz-sp5-tray-100-000001148-p1149737854/

Vendor sites:
-   https://smicro.hu/amd-socket-sp5-5
-   https://www.senetic.hu/category/amd-cpu-epyc-9004-11151/
-   https://kontaktor.hu/muszaki_cikkek_357/szamitastechnika_1797/alkatreszek_11508/processzorok_11635/amd_epyc_11688
-   https://www.arukereso.hu/processzor-c3139/f:tdp=0-350,amd-socket-sp5,amd-epyc/?orderby=1
-   Motherboards:
    -   https://smicro.hu/amd-sp5-5?filtrPriceFrom=&filtrPriceTo=&filter%5B2294%5D%5B%5D=39137&filter%5B2424%5D%5B%5D=42927&filter%5B2317%5D%5B%5D=38124&filter%5B2316%5D%5B%5D=38705&filter%5B2316%5D%5B%5D=39193&filter%5B2315%5D%5B%5D=40251&filter%5B2315%5D%5B%5D=43437&filter%5B2360%5D%5B%5D=39223

Prices:
-   CPU:
-   CPU cooler:
-   RAM:
    -   DDR5 RDIMM 1Rx4 or 2Rx8
    -   12 x [Samsung 32GB DDR5 4800MHz M323R4GA3BB0-CQK](https://www.arukereso.hu/memoria-modul-c3577/samsung/32gb-ddr5-4800mhz-m323r4ga3bb0-cqk-p818973822/): 12 x 150 = 1800 EUR
    -   For 2-CPU architecture:
        -   24 x [Kingston 16GB DDR5 4800MHz KSM48E40BS8KI-16HA](https://www.arukereso.hu/memoria-modul-c3577/kingston/16gb-ddr5-4800mhz-ksm48e40bs8ki-16ha-p1054408474/): 24 x 100 = 2400 EUR
-   SSD:
    -   [Samsung 990 PRO 4TB (MZ-V9P4T0BW)](https://belso-ssd-meghajto.arukereso.hu/samsung/990-pro-4tb-mz-v9p4t0bw-p1002242350/): 300 EUR
-   Motherboard:
    -   CEB:
        -   [Asus K14PA-U12](https://smicro.hu/asus-k14pa-u12-90sb0ci0-m0uay0-4?aku=db3621a52f6055ee636a6fee6ff8a353): 800 EUR
        -   [GENOAD8QM3‑2T/BCM](https://smicro.hu/asrock-rack-genoad8qm3-2t-bcm-5): 1700 EUR
    -   ATX:
        -   [Supermicro MBD-H13SSL-NT-O](https://smicro.hu/supermicro-mbd-h13ssl-nt-o-4): 830 EUR
        -   [GIGABYTE MZ33-AR0](https://www.arukereso.hu/alaplap-c3128/gigabyte/mz33-ar0-p1005435430/): 1100 EUR
-   PSU
-   Case
-   Total price without GPU:

</details> <!-- Server prices -->

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
