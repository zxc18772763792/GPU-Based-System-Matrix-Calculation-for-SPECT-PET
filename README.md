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

# Feel Free to Contact Me :)
I test this program with self-collimating SPECT system, this analytical system matrix matched perfect with experimental results. Calculation of photon electric system matrix with 160*160 2D FOV and 5632 crystals on a RTX 6000 Ada GPU only costs 3 mins. Calculation of primary compton scatter system matrix only costs 60 mins.
There are still some hard codes, but I'm too busy to fix these problems, sorry about that. :( If you have any questions, feel free to contact me. :)

E-mai: 18772763792@163.com, zhengxc21@mails.tsinghua.edu.cn
Wechat: zxc18772763792
