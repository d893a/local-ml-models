# Technical Details: [George Hotz AMD PC Build for tinygrad](https://www.youtube.com/watch?v=IhkXxFJ_qeI)

## Hardware Components

### Motherboard
- **Model**: ASRock Rack Rome a2t/BCM
- **Manufacturer**: ASRock (described as "ASUS's professional data center division")
- **Features**: 7 full bandwidth PCIe ports
- **Price**: $649 retail (mentioned buying another one for $610 on eBay)

### CPU
- **Model**: AMD EPYC 7662 64-core processor
- **Generation**: Gen 2 EPYC
- **Original Price**: $6,000 CPU from 2019
- **Purchase Price**: $800

### RAM
- **Total**: 64GB
- **Manufacturer**: Samsung (made in Philippines)
- **Configuration**: 8GB sticks, single rank
- **Speed**: 3200 MHz
- **Price**: ~$38 per stick, total around $300
- **Note**: George regretted not buying more RAM during software building

### GPUs
- **Model**: 2x AMD Radeon 7900 XTX (RDNA3 architecture)
- **Memory**: 24GB per GPU
- **Performance**: 13 teraflops measured during testing
- **Connection**: Using PCIe extenders (16x PCIe4)
- **Issues**: One GPU not connecting properly (likely due to longer extender)
- **Extender lengths**: One 30cm, one 15cm

### Storage
- **Model**: Samsung SSD 990 Pro 2TB NVMe

### Power Supply
- **Primary**: EVGA 1600 watt
- **Backup**: Dell server power supplies (2000W at 220V, limited to 1000W on US power)
- **Alternative**: Mentioned Dell 2400W power supplies that can do 1400W

### Case
- Bitcoin mining case

### Cooling
- Noctua fans (described as "12 volt fans run at 5 volts")

### Network
- Dual 10 gigabit network (on PCIe)

## Software/OS

- **Operating System**: Ubuntu Server 22.04 "Jammy Jellyfish"
- **ML Framework**: tinygrad
- **Model Testing**: LLAMA
- **Drivers**: ROCm stack for AMD GPUs
- **Issues**: Required kernel update to support RDNA3 GPUs

## Purpose of Build

- Building a training library competitive with PyTorch on AMD GPUs
- Goal to get AMD on MLPerf benchmark by end of year
- Running LLAMA AI models (mentioned 7GB and 13GB versions)
- Planning for model parallel and data parallel training

## Technical Advantages/Challenges

### AMD Advantages
- Fully documented RDNA3 architecture (vs. NVIDIA's partial documentation)
- AMD consumer GPUs support peer-to-peer communication (NVIDIA consumer GPUs don't)

### Challenges
- Difficult ROCm driver installation process
- Limited documentation
- Compatibility issues with kernel version
- PCIe extender reliability

## Total Cost
- Approximately $5,000 total for the build

## Future Plans
- Add more GPUs (planning for 6 total)
- Add 200 gigabit network connection for joint training
- Run 13GB LLAMA in float16 across multiple cards
- Explore model parallelism and data parallelism

---

# George Hotz AMD PC Build Stream Summary

This stream documents George Hotz (geohot) building a high-performance AMD-based computer specifically for running tinygrad, his neural network framework designed for simplicity and wide hardware compatibility.

## Hardware Configuration
- **CPU**: AMD EPYC 7662 (64-core) - purchased for $800 (originally a $6,000 CPU from 2019)
- **GPUs**: Two AMD Radeon RX 7900 XTX (24GB VRAM each)
- **Motherboard**: ASRock Rack Rome D8-2T with 7 full bandwidth PCIe slots
- **Memory**: 64GB RAM (which proved insufficient during compilation)
- **Storage**: Samsung 990 Pro 2TB NVMe SSD
- **Power Supply**: EVGA 1600W (also purchased Dell 2000W server PSUs)
- **Case**: Repurposed Bitcoin mining case with PCIe extenders

## The AMD Challenge
A significant portion of the stream involves Hotz struggling with AMD's complex software stack:
1. Initial attempts to install ROCm drivers failed
2. Had to build numerous components from source (HSA runtime, Rock-T, Rock-R, Rock-M, etc.)
3. Ran into RAM limitations during compilation
4. Eventually discovered the main issue was an outdated Linux kernel
5. Successfully got one GPU working, but had issues with the second (likely due to PCIe extender)

## Why AMD Over NVIDIA?
Hotz explicitly states: "One of the goals of the tiny Corp is fucking NVIDIA" because:
- AMD provides complete documentation for RDNA3 architecture
- AMD GPUs support direct peer-to-peer communication between cards
- NVIDIA artificially limits consumer GPU capabilities
- Hotz wants to create competition in the ML hardware space

## Project Goals
- Make tinygrad run efficiently on AMD hardware
- Get AMD on ML performance benchmarks by end of year
- Eventually support 6 GPUs in the system
- Run 13GB LLaMA models across two cards at reasonable speed
- Create a training library competitive with PyTorch on AMD GPUs

The stream ends with Hotz successfully running a basic tinygrad test on one GPU and beginning to load LLaMA weights, but still needing to troubleshoot why the second GPU isn't being detected.

This build represents Hotz's commitment to creating an alternative to NVIDIA's dominance in machine learning hardware, despite the significant technical challenges involved in using AMD's less mature ML ecosystem.