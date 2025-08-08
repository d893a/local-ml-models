# Hardware alternatives for running machine learning models locally

The following alternatives are detailed:
-   [Mini PC with mobile CPU + NPU](#mini-pc-with-mobile-cpu--npu)
-   Desktop PC
-   Workstation
-   Server

## Mini PC with mobile CPU + NPU

Expected performance for a 10 GB LLM model:
-   Prompt processing: 84 token/s
-   Token generation: 11 toen/s

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

<div class="page"/>

## Desktop PC

Expected CPU-only performance for a 10 GB LLM model:
-   Prompt processing: 125 token/s
-   Token generation: 8 token/s

Supports 1 CPIe x16 GPU, or 2 CPIe x8 GPUs

### AMD Ryzen 9 9950X3D

Specs:
-   CPU: AMD Ryzen 9 9950X3D
    ([Wikipedia](https://en.wikipedia.org/wiki/List_of_AMD_Ryzen_processors#Granite_Ridge_(9000_series,_Zen_5_based)))
    ([AMD](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html))
    ([TechPowerUp](https://www.techpowerup.com/cpu-specs/ryzen-9-9950x3d.c3993))
    -   128GB L3 cache
-   Chipsets
    ([AMD doc](https://www.amd.com/en/products/processors/chipsets/am5.html))
    ([Image](assets/amd-am5.png))
    ([TechPowerUp](https://www.techpowerup.com/cpu-specs/ryzen-9-9950x3d.c3993#gallery-5)):
    -   X870E:
        -   USB 4.0
        -   2 SUPERSPEED USB 20Gbps
        -   PCIe 5.0 1x16 or 2x8
        -   1x4 PCIe 5.0 plus 4x PCIe GPP
-   RAM:
-   Motherboard:
    -   ASRock X870E Taichi
    -   Ausus ROG Crosshair X870E Hero
    -   Biostar X870E Valkyrie
    -   Gigabyte X870E Aorus Xtreme AI
    -   MSI Meg X870E Godlike
-   PSU:
-   Case:

Desktop CPU Links:
-   https://www.anandtech.com/show/21524/the-amd-ryzen-9-9950x-and-ryzen-9-9900x-review/10
-   https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html
-   https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html
-   https://www.amd.com/en/products/processors/chipsets/am5.html
-   https://skatterbencher.com/2025/03/11/skatterbencher-85-ryzen-9-9950x3d-overclocked-to-5900-mhz/
