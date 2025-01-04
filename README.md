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
   ./ScatterGen -PE <path_to_PE_SystemMatrix> 
                -GeoCrystal <path_to_CrystalGeometryRelationship>
                -GeoCollimator <path_to_CollimatorGeometryRelationship>
                -cuda <cuda_device_id>
```
