#define _CRT_SECURE_NO_WARNINGS
 
// PEGen_CircularHole.cpp
// Description:
//   This program calculates the photon-electric system matrix.
//   Fixed for integer overflow with large detector*voxel counts.
//
// Usage:
//   ./PE_SysMat_Gen -cuda <cuda_device_id>
 
#include <fstream> 
#include <stdio.h>  
#include <stdlib.h>
#include <math.h>
#include <time.h>   
#include <iostream>
#include <cstring>
#include <string> 
#include <vector>
#include <thread>
#include <algorithm>
 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
 
#include "PESysMatGen.h"
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
	
	int numCollimatorLayers = (int)floor(parameter_Collimator[0]+0.001f);
	float FOV2Collimator0 = parameter_Image[11];
	for (int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayers; id_CollimatorLayer++)
	{
		cout << "############ Collimator " << id_CollimatorLayer << " ############" << endl;
		cout << "Number of collimator holes = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0] << endl;	
		cout << "Width of collimator layer(X direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1] << "mm" << endl;
		cout << "Thickness of collimator layer(Y direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2] << "mm" << endl;
		cout << "Height of collimator layer(Z direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3] << "mm" << endl;
		cout << "Collimator Layer to 1st Collimator Layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 4] << "mm" << endl;
		cout << "Total Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5] << endl;
		cout << "Photon-electric Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 6] << endl;
		cout << "Compton Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 7] << endl;
	}
 
	cout << "FOV center to 1st Collimator = " << FOV2Collimator0 << endl;
	////////////////////////////////////////////////////
	vector<int> cuda_ids;
 
	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-cuda") == 0 && i + 1 < argc)
		{
			appendCudaIds(argv[i + 1], cuda_ids);
			i++;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << "Usage: " << argv[0] << " [-cuda GPU_ID[,GPU_ID...]]" << endl;
			return 0;
		}
		else
		{
			cerr << "Unknown parameter or missing argument: " << argv[i] << endl;
			cout << "Usage: " << argv[0] << " [-cuda GPU_ID[,GPU_ID...]] " << endl;
			return EXIT_FAILURE;
		}
	}
	if (cuda_ids.empty()) cuda_ids.push_back(0);
 
	////////////////////////////////////////////////////
	size_t numProjectionsingle = (size_t)floor(parameter_Detector[0]+0.001f);
 
	int numImageVoxelX = (int)floor(parameter_Image[0] + 0.001f);
	int numImageVoxelY = (int)floor(parameter_Image[1] + 0.001f);
	int numImageVoxelZ = (int)floor(parameter_Image[2] + 0.001f);
	float widthImageVoxelX = parameter_Image[3];
	float widthImageVoxelY = parameter_Image[4];
	float widthImageVoxelZ = parameter_Image[5];
	int numRotation_ = (int)floor(parameter_Image[6]+0.001);
	float angelPerRotation = parameter_Image[7];
	float shiftFOVX= parameter_Image[8];
	float shiftFOVY = parameter_Image[9];
	float shiftFOVZ = parameter_Image[10];
 
	const size_t numProjectionSingle = numProjectionsingle;
	const size_t numImagebin = (size_t)numImageVoxelX * numImageVoxelY * numImageVoxelZ;
	const size_t numRotation = numRotation_;
	const size_t totalElements = numProjectionSingle * numImagebin * numRotation;
 
	printf("Allocating output: %zu elements (%.2f GB)\n", totalElements, (double)totalElements * 4.0 / 1e9);
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
 
		const size_t workerCount = min(cuda_ids.size(), numImagebin);
		vector<thread> workers;
		vector<int> results(workerCount, 0);
		size_t base = numImagebin / workerCount;
		size_t rem = numImagebin % workerCount;
		size_t imageBinStart = 0;
		for (size_t worker = 0; worker < workerCount; ++worker)
		{
			size_t imageBinCount = base + (worker < rem ? 1 : 0);
			int cuda_id = cuda_ids[worker];
			workers.emplace_back([&, worker, imageBinStart, imageBinCount, cuda_id]() {
				float localImage[100];
				memcpy(localImage, parameter_Image, sizeof(float) * 100);
				localImage[20] = float(idxRotation);
				results[worker] = PESysMatGen(parameter_Collimator, parameter_Detector, localImage,
					out + idxRotation * numProjectionSingle * numImagebin, cuda_id,
					imageBinStart, imageBinCount);
			});
			imageBinStart += imageBinCount;
		}
		for (auto& workerThread : workers) workerThread.join();
		for (size_t worker = 0; worker < workerCount; ++worker)
		{
			if (results[worker] < 0)
			{
				cerr << "GPU worker " << worker << " failed on rotation " << idxRotation << endl;
				delete[] out;
				return EXIT_FAILURE;
			}
		}
		int q = results.empty() ? 0 : results[0];
 
		printf("numImagebin = %d\n", q);
	}
 
	char Fname[2048];
	sprintf(Fname, "PE_SysMat_shift_%f_%f_%f_v3.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
	FILE* fp1;
	fp1 = fopen(Fname, "wb+");
	if (fp1 == 0) { puts("error"); exit(0); }
	fwrite(out, sizeof(float), totalElements, fp1);
	fclose(fp1);
 
	delete[] out;
 
	cout << "########################" << endl;
	cout << "Photon Electric Sysmat Written." << endl;
	cout << "########################" << endl;
	return 0;
}
 
