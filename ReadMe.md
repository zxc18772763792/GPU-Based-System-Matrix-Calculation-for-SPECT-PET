# GPU-Based System Matrix Calculation for SPECT/PET

A GPU-accelerated system matrix calculation tool designed for SPECT and PET systems, particularly those with complex geometries. This homemade software has been tested and performs efficiently on my systems.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Calculate Photon-Electric System Matrix](#calculate-photon-electric-system-matrix)
  - [Calculate Primary Compton System Matrix](#calculate-primary-compton-system-matrix)
  - [(Optional) Calculate Inter-Crystal Primary Compton System Matrix](#optional-calculate-inter-crystal-primary-compton-system-matrix)
  - [Monte Carlo System Matrix on Multiple GPUs (140.5 keV / 511 keV)](#monte-carlo-system-matrix-on-multiple-gpus-1405-kev--511-kev)
  - [Element Visualizer](#element-visualizer)
- [Parameter Files](#parameter-files)
  - [Param_Collimator.dat](#param_collimatordat)
  - [Param_Detector.dat](#param_detectordat)
  - [Param_Image.dat](#param_imagedat)
  - [Param_Physics.dat](#param_physicsdat)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This project provides a GPU-based framework for calculating system matrices essential for SPECT (Single Photon Emission Computed Tomography) and PET (Positron Emission Tomography) systems. It is optimized for complex geometrical configurations and leverages GPU acceleration to significantly reduce computation time.

## Features

- **GPU Acceleration:** Utilizes CUDA for high-performance matrix calculations.
- **Flexible Geometry Support:** Designed to handle complex system geometries.
- **Modular Structure:** Separate modules for photon-electric and Compton scatter calculations.
- **Configurable Parameters:** Easily adjustable parameter files to define system configurations.
- **Multi-GPU Monte Carlo:** `MCGen_MultiGPU` simulates full photon transport (photoelectric + K-fluorescence, Klein–Nishina Compton, Rayleigh, energy-dependent attenuation, energy-resolution windows) for 140.5 keV and 511 keV from the same parameter files, scaling over any number of GPUs.
- **Element Visualizer:** a zero-dependency browser UI to inspect collimator layers, holes, crystals, the FOV, system-matrix rows/columns/sensitivity maps and Monte Carlo spectra and photon tracks.

## Prerequisites

- **CUDA Toolkit:** Ensure that the CUDA toolkit is installed and properly configured on your system.
- **Compiler:** A compatible C++ compiler (e.g., `gcc`, `clang`).
- **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 6000 Ada).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/zxc18772763792/GPU-Based-System-Matrix-Calculation-for-SPECT-PET.git
   cd GPU-Based-System-Matrix-Calculation-for-SPECT-PET
   ```

2. **Prepare Parameter Files:**

   Before compiling, prepare four parameter files (`Param_Collimator.dat`, `Param_Detector.dat`, `Param_Image.dat`, `Param_Physics.dat`) as described in the [Parameter Files](#parameter-files) section.

## Usage

### Calculate Photon-Electric System Matrix

1. **Compile the Photon-Electric Module:**

   Navigate to the `PE_Gen_RayTracing_CircularHole` directory and compile the code.

   ```bash
   cd PE_Gen_RayTracing_CircularHole
   ./bd
   ```

2. **Run the Photon-Electric System Matrix Generator:**

   ```bash
   ./PESysMatGen -cuda 0
   ```

   Replace `0` with the appropriate CUDA device ID if necessary.

### Calculate Primary Compton System Matrix

1. **Compile the Compton Scatter Module:**

   Navigate to the `ScatterGen_RayTracing_CircularHole` directory and compile the code.

   ```bash
   cd ../ScatterGen_RayTracing_CircularHole
   ./bd
   ```

2. **Run the Primary Compton Scatter System Matrix Generator:**

   ```bash
   ./ScatterGen_CircularHole \
     -PE <path_to_PE_SystemMatrix> \
     -GeoCrystal <path_to_CrystalGeometryRelationship> \
     -GeoCollimator <path_to_CollimatorGeometryRelationship> \
     -cuda <cuda_device_id> \
     [-components] [-model v2|legacy]
   ```

   Replace placeholders with the actual paths and CUDA device ID. With `-components` the crystal-scatter
   and collimator-scatter parts are also written separately (`Scatter_Crystal_SysMat_shift_*.sysmat`,
   `Scatter_Collimator_SysMat_shift_*.sysmat`). The PE file name written by PEGen ends in `_v3.sysmat`,
   so pass it with `-PE`.

   The default scatter model (`-model v2`) integrates the single Compton scatter over the volume where
   it happens: over the whole scatter crystal (weighted by the transmission of the incoming beam through
   the collimator and the crystals to every point) for inter-crystal scatter, and over a thin shell around every collimator hole plus the rest of the plate
   for collimator scatter, with exact solid angles of the crystal faces, Klein–Nishina angles and
   energies, and attenuation of the scattered photon in the crystals and the collimator. Against the
   first-order Monte Carlo of `MCGen_MultiGPU` (`physics = single_scatter`, same physics definition) it
   agrees within 3 % for crystal scatter and 5 % for collimator scatter at 140.5 keV (1 % for both at
   511 keV), photopeak and wide energy windows, see
   [`examples/analytic_vs_mc`](examples/analytic_vs_mc/ReadMe.md). `-model legacy` runs the previous
   kernels (one scatter point per crystal or hole), which were off by factors of 0.002–24 in that example.

### (Optional) Calculate Inter-Crystal Primary Compton System Matrix

1. **Compile the Inter-Crystal Compton Module:**

   Navigate to the `ScatterGen_Crystal` directory and compile the code.

   ```bash
   cd ../ScatterGen_Crystal
   ./bd
   ```

2. **Run the Inter-Crystal Compton System Matrix Generator:**

   ```bash
   ./ScatterGen_Crystal \
     -PE <path_to_PE_SystemMatrix> \
     -GeoCrystal <path_to_CrystalGeometryRelationship> \
     -cuda <cuda_device_id>
   ```

### Monte Carlo System Matrix on Multiple GPUs (140.5 keV / 511 keV)

`MCGen_MultiGPU` reads the same four parameter files and writes system matrices with the same
`float32 [rotation][detector][voxel]` layout as the analytic modules. Voxel chunks are distributed
dynamically over all GPUs; every photon history has its own Philox random stream, so results do not
depend on the number of GPUs (and rotation ranges can be split across machines). In
`physics = primary_pe` mode it reproduces the analytic photoelectric matrix element by element
(z-score rms ≈ 1 at 140.5 keV and 511 keV); in full mode it adds septal scatter, multiple Compton
scattering, Rayleigh scattering, K x-rays and inter-crystal scatter, and can output the
primary / collimator-scatter / inter-crystal-scatter components and per-element statistical errors.

```bash
cd MCGen_MultiGPU
./bd                                   # or: make   (make cpu builds a CPU-only binary)
cd /path/to/params
/path/to/MCGen_MultiGPU/MCGen_MultiGPU /path/to/MCGen_MultiGPU/MC_Config_Tc99m.txt -gpus 0,1,2,3
/path/to/MCGen_MultiGPU/MCGen_MultiGPU /path/to/MCGen_MultiGPU/MC_Config_511keV.txt -histories_per_voxel 1e6
```

See [MCGen_MultiGPU/ReadMe.md](MCGen_MultiGPU/ReadMe.md) for physics, variance reduction, all
configuration keys, the material database (W, Pb, W alloys, Au, Pt, NaI, CsI, LYSO, BGO, GAGG, CZT,
LaBr3, …) and the validation results. `tools/make_demo_params.py` writes a small demo geometry and
`tools/compare_sysmat.py` compares two system matrices (e.g. analytic vs Monte Carlo).

### Analytic vs Monte Carlo example

[`examples/analytic_vs_mc`](examples/analytic_vs_mc/ReadMe.md) runs PEGen + ScatterGen and
MCGen_MultiGPU on one GPU for a small multi-pinhole system at 140.5 keV and 511 keV and compares the
photoelectric, crystal-scatter and collimator-scatter components (`./run_example.sh tc99m gpu`).
Without a GPU, `./run_example.sh tc99m cpu` runs the original analytic CUDA kernels compiled for the
CPU ([`tools/analytic_cpu`](tools/analytic_cpu/build.py)) and the CPU build of the Monte Carlo code.

### Element Visualizer

Open `Visualizer/index.html` directly, or serve a run directory (recommended for large matrices on a
remote GPU server, via SSH port forwarding):

```bash
python3 Visualizer/serve.py --data /path/to/run_dir     # then open http://127.0.0.1:8765
```

It shows the collimator plates, holes, crystals (coloured by attenuation, energy resolution, flag,
system-matrix projection, …) and the FOV in 3D with picking and section planes, an x–z layout map,
system-matrix rows / columns / sensitivity maps streamed from disk, Monte Carlo energy spectra and
photon tracks, and a list of geometry consistency checks. See [Visualizer/ReadMe.md](Visualizer/ReadMe.md).

## Parameter Files

Four parameter files are required to define your system. Each file is a pure `float32` array organized as follows:

### `Param_Collimator.dat`

Defines the collimator configuration.

- **Index 0:** `numCollimatorLayers` — Number of collimator layers.
- **For each Collimator Layer (`id_CollimatorLayer`):**
  - `[id * 10 + 0]:` Number of Collimator Holes.
  - `[id * 10 + 1]:` Width of the Collimator Layer (mm).
  - `[id * 10 + 2]:` Thickness of the Collimator Layer (mm).
  - `[id * 10 + 3]:` Height of the Collimator Layer (mm).
  - `[id * 10 + 4]:` Distance between the 1st and current collimator layer (mm).
  - `[id * 10 + 5]:` Total Attenuation Coefficient.
  - `[id * 10 + 6]:` Photon-Electric (PE) Attenuation Coefficient.
  - `[id * 10 + 7]:` Compton Attenuation Coefficient.
- **For each Hole (`id_Hole`):**
  - `[id_Hole * 9 + 100]:` X-coordinate of Hole Center.
  - `[id_Hole * 9 + 101]:` Y1-coordinate of Hole Center.
  - `[id_Hole * 9 + 102]:` Y2-coordinate of Hole Center.
  - `[id_Hole * 9 + 103]:` Z-coordinate of Hole Center.
  - `[id_Hole * 9 + 104]:` Radius of Hole.
  - `[id_Hole * 9 + 105]:` Total Attenuation Coefficient of Hole.
  - `[id_Hole * 9 + 106]:` PE Attenuation Coefficient of Hole.
  - `[id_Hole * 9 + 107]:` Compton Attenuation Coefficient of Hole.
  - `[id_Hole * 9 + 108]:` Flag.

### `Param_Detector.dat`

Defines the detector configuration.

- **Index 0:** `numDetectorBins` — Number of detector bins.
- **For each Detector (`id_Detector`):**
  - `[id * 12 + 1]:` X-coordinate of Detector Center.
  - `[id * 12 + 2]:` Y-coordinate of Detector Center (set Y of 1st collimator to 0).
  - `[id * 12 + 3]:` Z-coordinate of Detector Center.
  - `[id * 12 + 4]:` Width of Detector (mm).
  - `[id * 12 + 5]:` Thickness of Detector (mm).
  - `[id * 12 + 6]:` Height of Detector (mm).
  - `[id * 12 + 7]:` Total Attenuation Coefficient (excluding Rayleigh scatter).
  - `[id * 12 + 8]:` Photon-Electric (PE) Attenuation Coefficient.
  - `[id * 12 + 9]:` Compton Attenuation Coefficient.
  - `[id * 12 + 10]:` Energy Resolution at Target PE Energy.
  - `[id * 12 + 11]:` Rotation Angle of Detector (Y-axis) [0, 2π).
  - `[id * 12 + 12]:` Flag.

### `Param_Image.dat`

Defines the image voxel configuration.

- **Index 0:** `numImageVoxelX` — Number of image voxels along the X-axis.
- **Index 1:** `numImageVoxelY` — Number of image voxels along the Y-axis.
- **Index 2:** `numImageVoxelZ` — Number of image voxels along the Z-axis.
- **Index 3:** `widthImageVoxelX` (mm) — Width of each voxel along the X-axis.
- **Index 4:** `widthImageVoxelY` (mm) — Width of each voxel along the Y-axis.
- **Index 5:** `widthImageVoxelZ` (mm) — Width of each voxel along the Z-axis.
- **Index 6:** `numRotation` — Number of rotations.
- **Index 7:** `anglePerRotation` (0~2π) — Angle increment per rotation.
- **Index 8:** `shiftFOVX` (mm) — Shift of the Field of View (FOV) along the X-axis.
- **Index 9:** `shiftFOVY` (mm) — Shift of the FOV along the Y-axis.
- **Index 10:** `shiftFOVZ` (mm) — Shift of the FOV along the Z-axis.
- **Index 11:** `FOV2Collimator0` (mm) — Distance from FOV to Collimator layer 0.

### `Param_Physics.dat`

Defines the physics parameters for the simulation.

- **Index 0:** `flagUsingCompton` — Enable (1) or disable (0) Compton scattering.
- **Index 1:** `flagSavingPESysmat` — Enable (1) or disable (0) saving PE system matrix.
- **Index 2:** `flagSavingComptonSysmat` — Enable (1) or disable (0) saving Compton system matrix.
- **Index 3:** `flagSavingPEComptonSysmat` — Enable (1) or disable (0) saving combined PE and Compton system matrix.
- **Index 4:** `flagUsingSameEnergyWindow` — Use (1) or not (0) the same energy window.
- **Index 5:** `lowerThresholdEnergyWindow` — Lower threshold of the energy window.
- **Index 6:** `upperThresholdEnergyWindow` — Upper threshold of the energy window.
- **Index 7:** `targetPEEnergy` — Target PE energy.
- **Index 8:** `flagCalculateCrystalGeometryRelationship` — Enable (1) or disable (0) calculation of crystal geometry relationship.
- **Index 9:** `flagCalculateCollimatorGeometryRelationship` — Enable (1) or disable (0) calculation of collimator geometry relationship.

## Contact

This program has been tested with an advanced SPECT system, and the analytical system matrix matches perfectly with experimental results. Calculating the photon-electric system matrix for a 200x200 2D FOV with ~6000 crystals on an RTX 6000 Ada GPU takes approximately **3 minutes**, while calculating the primary Compton scatter system matrix takes about **60 minutes**.

**Note:** There are some hard-coded elements in the current version. I apologize for any inconvenience and plan to address these issues when time permits. If you have any questions or need assistance, feel free to reach out.

- **Email:** [18772763792@163.com](mailto:18772763792@163.com) or [zhengxc21@mails.tsinghua.edu.cn](mailto:zhengxc21@mails.tsinghua.edu.cn)
- **WeChat:** zxc18772763792

## Acknowledgements

Thank you for using this tool. Contributions and feedback are welcome to help improve its functionality and performance.


## Usage Terms

This software is available **strictly for academic, research, and educational purposes only**. Commercial use is expressly prohibited without prior written permission. 

When using this work in academic publications or research, you must include proper attribution:

> [Xingchun Zheng, etc.], "GPU-Based System Matrix Calculation for SPECT/PET", GitHub repository, 2023. https://github.com/zxc18772763792/GPU-Based-System-Matrix-Calculation-for-SPECT-PET

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).


---

