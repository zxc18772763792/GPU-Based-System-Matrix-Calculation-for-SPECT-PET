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

## Prerequisites

- **CUDA Toolkit:** Ensure that the CUDA toolkit is installed and properly configured on your system.
- **Compiler:** A compatible C++ compiler (e.g., `gcc`, `clang`).
- **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 6000 Ada).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/GPU-Based-System-Matrix-Calculation-for-SPECT-PET.git
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
     -cuda <cuda_device_id>
   ```

   Replace placeholders with the actual paths and CUDA device ID.

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

This program has been tested with a self-collimating SPECT system, and the analytical system matrix matches perfectly with experimental results. Calculating the photon-electric system matrix for a 160x160 2D FOV with 5,632 crystals on an RTX 6000 Ada GPU takes approximately **3 minutes**, while calculating the primary Compton scatter system matrix takes about **60 minutes**.

**Note:** There are some hard-coded elements in the current version. I apologize for any inconvenience and plan to address these issues when time permits. If you have any questions or need assistance, feel free to reach out.

- **Email:** [18772763792@163.com](mailto:18772763792@163.com) or [zhengxc21@mails.tsinghua.edu.cn](mailto:zhengxc21@mails.tsinghua.edu.cn)
- **WeChat:** zxc18772763792

## Acknowledgements

Thank you for using this tool. Contributions and feedback are welcome to help improve its functionality and performance.

## License

*Specify the license under which your project is distributed, e.g., MIT, GPL, etc.*

---

*Feel free to customize this README further to better fit your project's specific needs and to include any additional sections or information as necessary.*
