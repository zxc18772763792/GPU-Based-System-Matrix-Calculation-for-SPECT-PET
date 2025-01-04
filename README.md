# GPU-Based-System-Matrix-Calculation-for-SPECT-PET
GPU-based system matrix calculation for SPECT and PET system, especially for complex geometry systems. Home made and worked well on my systems. :)
# Prepare
Before start the compile, please prepare 4 parameter files which define your system accroding to ReadMe.txt. These 4 parameter files are pure float32 arrays.
# Calculate Photon-Elecrtic System Matix
Complie the PE_Gen_RayTracing_CircularHole folder, you can refer to the compile command as ./PE_Gen_RayTracing_CircularHole/bd.
Run the programe PESysMatGen as:
```
./PESysMatGen -cuda 0
```
# Calculate Primary Compton System Matix
Complie the ScatterGen_RayTracing_CircularHole folder, you can refer to the compile command as ./ScatterGen_RayTracing_CircularHole/bd.
Run the programe ScatterGen_CircularHole as:
```
   ./ScatterGen_CircularHole -PE <path_to_PE_SystemMatrix> 
                -GeoCrystal <path_to_CrystalGeometryRelationship>
                -GeoCollimator <path_to_CollimatorGeometryRelationship>
                -cuda <cuda_device_id>
```

# (Optinal) Calculate only the inter-crystal Primary Compton System Matix
Complie the ScatterGen_Crystal folder, you can refer to the compile command as ./ScatterGen_Crystal/bd.
Run the programe ScatterGen_Crystal as:
```
   ./ScatterGen_Crystal -PE <path_to_PE_SystemMatrix> 
                -GeoCrystal <path_to_CrystalGeometryRelationship>
                -cuda <cuda_device_id>
```

# Parameter Files
4 Parameter files:Param_Collimator.dat; Param_Detector.dat; Param_Image.dat; Param_Physics.dat
They are all pure float32 array. Organized as fowolling:

Param_Collimator.dat:
Param_Collimator[0]:numCollimatorLayers
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0]:Number of Collimator Holes in Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1]:Width  of Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2]:Thickness  of Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3]:Height  of Collimator Layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 4]:Distance between1st collimator layer and collimator layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5]:Total Attneuation coefficient of collimator layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 6]:PE Attneuation coefficient of collimator layer id_CollimatorLayer
parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 7]:Compton Attneuation coefficient of collimator layer id_CollimatorLayer
parameter_Collimator[id_Hole * 9 + 100]:x of hole center
parameter_Collimator[id_Hole * 9 + 101]:y1 of hole center
parameter_Collimator[id_Hole * 9 + 102]:y2 of hole center
parameter_Collimator[id_Hole * 9 + 103]:z of hole center
parameter_Collimator[id_Hole * 9 + 104]:R of hole center
parameter_Collimator[id_Hole * 9 + 105]:Total Attneuation coefficient of hole
parameter_Collimator[id_Hole * 9 + 106]:PE Attneuation coefficient of hole
parameter_Collimator[id_Hole * 9 + 107]:Compton Attneuation coefficient of hole
parameter_Collimator[id_Hole * 9 + 108]:flag

Param_Detector.dat:
Param_Detector[0]:numDetectorBins
Param_Detector[id_Detector*12+1]:x of detector center
Param_Detector[id_Detector*12+2]:y of detector center (set y(1st collimator)=0)
Param_Detector[id_Detector*12+3]:z of detector center
Param_Detector[id_Detector*12+4]:width of detector
Param_Detector[id_Detector*12+5]:thickness of detector
Param_Detector[id_Detector*12+6]:height of detector
Param_Detector[id_Detector*12+7]:total attenuation coefficient of detector (without Rayleigh scatter)
Param_Detector[id_Detector*12+8]:photon-electrical attenuation coefficient of detector
Param_Detector[id_Detector*12+9]:compton attenuation coefficient of detector
Param_Detector[id_Detector*12+10]:energy resolution @ target PE energy
Param_Detector[id_Detector*12+11]:rotation angel of detector (y axis) [0,2pi)
Param_Detector[id_Detector*12+12]:flag

Param_Image.dat:
Param_Image[0]:numImageVoxelX
Param_Image[1]:numImageVoxelY
Param_Image[2]:numImageVoxelZ
Param_Image[3]:widthImageVoxelX(mm)
Param_Image[4]:widthImageVoxelY(mm)
Param_Image[5]:widthImageVoxelZ(mm)
Param_Image[6]:numRotation
Param_Image[7]:angelPerRotation(0~2pi)
Param_Image[8]:shiftFOVX(mm)
Param_Image[9]:shiftFOVY(mm)
Param_Image[10]:shiftFOVZ(mm)
Param_Image[11]:FOV2Collimator0(mm)

Param_Physics.dat:
Param_Physics[0]:flagUsingCompton
Param_Physics[1]:flagSavingPESysmat
Param_Physics[2]:flagSavingComptonSysmat
Param_Physics[3]:flagSaving PE+Compton Sysmat
Param_Physics[4]:flagUsingSameEnegryWindow
Param_Physics[5]:lowerThresholdofEnegryWindow
Param_Physics[6]:upperThresholdofEnegryWindow
Param_Physics[7]:target PE Energy
Param_Physics[8]:flagCalCulateCrystalGeometryRelationShip
Param_Physics[9]:flagCalCulateCollimatorGeometryRelationShip

# Feel Free to Contact Me :)
I test this program with self-collimating SPECT system, this analytical system matrix matched perfect with experimental results. Calculation of photon electric system matrix with 160*160 2D FOV and 5632 crystals on a RTX 6000 Ada GPU only costs 3 mins. Calculation of primary compton scatter system matrix only costs 60 mins.
There are still some hard codes, but I'm too busy to fix these problems, sorry about that. :( If you have any questions, feel free to contact me. :)

E-mai: 18772763792@163.com, or zhengxc21@mails.tsinghua.edu.cn

Wechat: zxc18772763792
