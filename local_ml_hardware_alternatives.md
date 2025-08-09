# Hardware alternatives for running machine learning models locally

-   [Embedded / mobile CPU](#embedded--mobile-cpu)
-   [Desktop PC](#desktop-pc)
-   Workstation
-   Server

The following GPU alternatives are listed:
-   NVIDIA RTX PRO 6000 Blackwell (96GB)
-   NVIDIA GeForce RTX 5090 (32GB)
-   NVIDIA GeForce RTX 5070 Ti SUPER (24GB)
-   NVIDIA GeForce RTX 5080 SUPER (24GB)
-   AMD Radeon RX 7900 XTX (24GB)

## Embedded / mobile CPU

<details><summary>Details</summary>

Expected performance for a 10 GB LLM model:
-   Prompt processing: 84 token/s
-   Token generation: 11 token/s

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
    -   4200 USD = 3600 EUR

### Other options

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

Specs:
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
        -   Maybe with 192MB L3 cache [when available](https://www.techpowerup.com/review/future-hardware-releases/#ryzenx3ddualcc)
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
    -   CPU max RAM: 192 GB
    -   Motherboard: 2x/4x DDR5 DIMM slots
    -   [128 GB](https://www.arukereso.hu/memoria-modul-c3577/f:kapacitas-128-gb,tipus-pc-memoria/?orderby=1): 360 EUR
        -   RAM BW upper limit: 2 x 8 x 5.6 = 89.6 GB/s
    -   [192 GB](https://www.arukereso.hu/memoria-modul-c3577/f:kapacitas-192-gb,tipus-pc-memoria/): 760 EUR
        -   [RAM BW upper limit](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html):
            2 channels x 8 x 3.6 MT/s = 57.6 GB/s
            -   ❗**Note:** 192 GB (vs 128 GB) system RAM will **reduce** 10 GB
                LLM model CPU-only token generation speed from 8 token/s down
                to 5 token/s maximum. Overclocking may improve a few token/s,
                no substantial change.
            -   ❗**Note:** This practically limits the system RAM to 128 GB,
                which puts the **maximum combined GPU VRAM size to ~64 GB**.
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

Price:
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
-   61.39 FP32 TFLOPS

Price:
-   [Árukereső](https://www.arukereso.hu/videokartya-c3142/f:amd-radeon-video-chipset,rx-7900-xtx/?orderby=1)
    -   341,000 to 419,000 HUF (860 to 1100 EUR)

</details>
