#define _CRT_SECURE_NO_WARNINGS

// ScatterGen_CircularHole.cpp
// Description:
//   This program calculates the scatter system matrix, considering only primary Compton events.
//   It should be executed after the PE (photoelectric effect) system matrix generation.
//
// Usage:
//   ./ScatterGen_CircularHole -PE <path_to_PE_SystemMatrix> 
//                -GeoCrystal <path_to_CrystalGeometryRelationship>
//                -GeoCollimator <path_to_CollimatorGeometryRelationship>
//                -cuda <cuda_device_id>
//
// Author: Xingchun Zheng @ tsinghua university
// Last Modified: 2024/12/21
// Version: 1.0



#include <fstream> 
#include <stdio.h>  
#include <stdlib.h>

#include <cstring>
#include <string> 
#include <chrono> 
#include <math.h>
#include <time.h>   
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

#include<cuda_runtime.h>
#include <device_launch_parameters.h>

#include "scatter.h"

using namespace std;

static void appendCudaIds(const char* text, vector<int>& cuda_ids)
{
	string item;
	for (const char* p = text; ; ++p)
	{
		if (*p == ',' || *p == '\0')
		{
			if (!item.empty())
			{
				cuda_ids.push_back(atoi(item.c_str()));
				item.clear();
			}
			if (*p == '\0') break;
		}
		else
		{
			item.push_back(*p);
		}
	}
}

int main(int argc, char* argv[])
{
	float* parameter_Collimator = new float[80000]();
	float* parameter_Detector = new float[80000]();
	float* parameter_Image = new float[100]();
	float* parameter_Physics = new float[100]();

	FILE* fid;
	fid = fopen("Params_Collimator.dat", "rb");
	fread(parameter_Collimator, sizeof(float), 80000, fid);
	fclose(fid);

	FILE* fid1;
	fid1 = fopen("Params_Detector.dat", "rb");
	fread(parameter_Detector, sizeof(float), 80000, fid1);
	fclose(fid1);
	
	FILE* fid2;
	fid2 = fopen("Params_Image.dat", "rb");
	fread(parameter_Image, sizeof(float), 100, fid2);
	fclose(fid2);

	FILE* fid3;
	fid3 = fopen("Params_Physics.dat", "rb");
	fread(parameter_Physics, sizeof(float), 100, fid3);
	fclose(fid3);
	////////////////////////////////////////////////////
	
	int numCollimatorLayers = (int)parameter_Collimator[0];
	float FOV2Collimator0 = parameter_Image[11];
	for (int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayers; id_CollimatorLayer++)
	{
		cout << "############ Collimator " << id_CollimatorLayer << " ############" << endl;
		cout << "Number of collimator holes = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0] << endl;	
		cout << "Width of collimator layer(X direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1] << "mm" << endl;
		cout << "Thickness of collimator layer(Y direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2] << "mm" << endl;
		cout << "Height of collimator layer(Z direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3] << "mm" << endl;
		cout << "Collimator Layer to 1st Collimator Layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 4] << "mm" << endl;
		cout << "Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5] << endl;

	}

	cout << "FOV center to 1st Collimator = " << FOV2Collimator0 << endl;

	////////////////////////////////////////////////////
	int numProjectionsingle =(int) parameter_Detector[0];

	int numImageVoxelX = (int)parameter_Image[0];
	int numImageVoxelY = (int)parameter_Image[1];
	int numImageVoxelZ = (int)parameter_Image[2];
	float widthImageVoxelX = parameter_Image[3];
	float widthImageVoxelY = parameter_Image[4];
	float widthImageVoxelZ = parameter_Image[5];
	int numRotation_ = (int)floor(parameter_Image[6]+0.001);
	float angelPerRotation = parameter_Image[7];
	float shiftFOVX= parameter_Image[8];
	float shiftFOVY = parameter_Image[9];
	float shiftFOVZ = parameter_Image[10];

	const size_t numProjectionSingle = (size_t)numProjectionsingle;
	const size_t numImagebin = (size_t)numImageVoxelX * numImageVoxelY * numImageVoxelZ;
	const size_t numRotation = (size_t)numRotation_;
	const size_t totalElements = numProjectionSingle * numImagebin * numRotation;

	string FnamePE;
	string FnameGeoCrystal = "GeometryRelationShip_Crystal2Crystal"; 
	string FnameGeoCollimator = "GeometryRelationShip_Collimator2Crystal";
	vector<int> cuda_ids;

	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-PE") == 0 && i + 1 < argc)
		{
			FnamePE = argv[i + 1];
			i++; 
		}
		else if (strcmp(argv[i], "-GeoCrystal") == 0 && i + 1 < argc)
		{
			FnameGeoCrystal = argv[i + 1];
			parameter_Physics[8] = 0;
			i++; 
		}
		else if (strcmp(argv[i], "-GeoCollimator") == 0 && i + 1 < argc)
		{
			FnameGeoCollimator= argv[i + 1];
			parameter_Physics[9] = 0;
			i++;
		}
		else if (strcmp(argv[i], "-cuda") == 0 && i + 1 < argc)
		{
			appendCudaIds(argv[i + 1], cuda_ids);
			i++;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path] [-GeoCollimator GeometryRelationShip_Collimator2Crystal_path] [-cuda GPU_ID[,GPU_ID...]]" << endl;
			return 0;
		}
		else
		{
			cerr << "Unknown parameter or missing argument: " << argv[i] << endl;
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path] [-GeoCollimator GeometryRelationShip_Collimator2Crystal_path] " << endl;
			return EXIT_FAILURE;
		}
	}
	if (cuda_ids.empty()) cuda_ids.push_back(0);

	//////////////////////////// PE SysMat Loading ///////////////////////////////
	float* PE_SysMat = new float[totalElements]();

	if (FnamePE.empty())
	{
		char bufferPE[2048];
		snprintf(bufferPE, sizeof(bufferPE), "PE_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FnamePE = bufferPE;
	}

	cout << "Photon Electric SysMat: " << FnamePE << endl;

	auto start_ioPE = std::chrono::high_resolution_clock::now();
	FILE* fp0;
	fp0 = fopen(FnamePE.c_str(), "rb");
	if (fp0 == 0) { puts("error"); exit(0); }
	fread(PE_SysMat, sizeof(float), totalElements, fp0);
	fclose(fp0);
	auto end_ioPE = std::chrono::high_resolution_clock::now();
	auto duration_ioPE = std::chrono::duration_cast<std::chrono::milliseconds>(end_ioPE - start_ioPE);
	cout << "Time of io PE System Matrix: " << duration_ioPE.count() << " ms" << endl;

	//////////////////////////// Scatter Function Start ///////////////////////////////
	cout << "Geometry RelationShip Crystal2Crystal:  " << FnameGeoCrystal << endl;
	cout << "Geometry RelationShip Collimator2Crystal:  " << FnameGeoCollimator << endl;

	auto start_scatter = std::chrono::high_resolution_clock::now();

	float* out = new float[totalElements]();

	printf("FOV dimension : %d %d %d\n", numImageVoxelX, numImageVoxelY, numImageVoxelZ);
	printf("FOV Voxel Size(mm) : %f %f %f\n", widthImageVoxelX, widthImageVoxelY, widthImageVoxelZ);
	cout << "Using " << cuda_ids.size() << " GPU worker(s), split by image bins." << endl;
	for (size_t idxRotation = 0; idxRotation < numRotation; idxRotation++)
	{
		cout << "########################" << endl;
		cout << "Rotation (" << idxRotation << ") processing ..." << endl;
		cout << "########################" << endl;


		cout << "Shift FOV in X = " << shiftFOVX << "mm" << endl;
		cout << "Shift FOV in Y = " << shiftFOVY << "mm" << endl;
		cout << "Shift FOV in Z = " << shiftFOVZ << "mm" << endl;

		const bool generatingGeometry = ((int)floor(parameter_Physics[8]) == 1 || (int)floor(parameter_Physics[9]) == 1);
		const size_t workerCount = generatingGeometry ? 1 : min(cuda_ids.size(), (size_t)numImagebin);
		vector<thread> workers;
		vector<int> results(workerCount, 0);
		size_t base = (size_t)numImagebin / workerCount;
		size_t rem = (size_t)numImagebin % workerCount;
		size_t imageBinStart = 0;
		for (size_t worker = 0; worker < workerCount; ++worker)
		{
			size_t imageBinCount = base + (worker < rem ? 1 : 0);
			int cuda_id = cuda_ids[worker];
			workers.emplace_back([&, worker, imageBinStart, imageBinCount, cuda_id]() {
				float localImage[100];
				float localPhysics[100];
				memcpy(localImage, parameter_Image, sizeof(float) * 100);
				memcpy(localPhysics, parameter_Physics, sizeof(float) * 100);
				localImage[20] = float(idxRotation);
				results[worker] = scatter(parameter_Collimator, parameter_Detector, localImage, localPhysics,
					PE_SysMat + idxRotation * numProjectionSingle * numImagebin,
					FnameGeoCrystal.c_str(), FnameGeoCollimator.c_str(),
					out + idxRotation * numProjectionSingle * numImagebin,
					cuda_id, imageBinStart, imageBinCount);
			});
			imageBinStart += imageBinCount;
		}
		for (auto& workerThread : workers) workerThread.join();
		if (generatingGeometry) {
			parameter_Physics[8] = 0;
			parameter_Physics[9] = 0;
		}
		for (size_t worker = 0; worker < workerCount; ++worker)
		{
			if (results[worker] < 0)
			{
				cerr << "GPU worker " << worker << " failed on rotation " << idxRotation << endl;
				delete[] out;
				delete[] PE_SysMat;
				return EXIT_FAILURE;
			}
		}
		int q = results.empty() ? 0 : results[0];

		printf("numImagebin = %d\n", q);
	}

	auto end_scatter = std::chrono::high_resolution_clock::now();
	auto duration_scatter = std::chrono::duration_cast<std::chrono::milliseconds>(end_scatter - start_scatter);
	cout << "Time of scatter function: " << duration_scatter.count()/1000.0/60.0 << " min" << endl;

	if (parameter_Physics[2] == 1)
	{
		char Fname[2048];
		sprintf(Fname, "Scatter_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FILE* fp1;
		fp1 = fopen(Fname, "wb+");
		if (fp1 == 0) { puts("error"); exit(0); }
		fwrite(out, sizeof(float), totalElements, fp1);
		fclose(fp1);

		cout << "########################" << endl;
		cout << "Compton Scatter Sysmat written." << endl;
		cout << "########################" << endl;
	}
	if (parameter_Physics[3] == 1) 
	{
		float* SysMat = new float[totalElements]();
		for (size_t i = 0; i < totalElements; i++) 
		{
			SysMat[i] = PE_SysMat[i] + out[i];
		}
		
		char Fname3[2048];
		sprintf(Fname3, "SysMat_withScatter_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FILE* fp2;
		fp2 = fopen(Fname3, "wb+");
		if (fp2 == 0) { puts("error"); exit(0); }
		fwrite(SysMat, sizeof(float), totalElements, fp2);
		fclose(fp2);

		cout << "########################" << endl;
		cout << "Full Sysmat written." << endl;
		cout << "########################" << endl;
	}
	return 0;
}

