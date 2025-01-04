#define _CRT_SECURE_NO_WARNINGS

// ScatterGen.cpp
// Calculate the scatter system matrix, with primary compton events only
// run this program after the PE system matrix generation
// usage: ./ScatterGen -PE "path/to/PE_SystemMatrix" -GeoCrystal "path/to/Scatter_SystemMatrix" -cuda int
// athour: xingchun zheng @ tsinghua university
// last modified: 2024/12/21
// version 1.0

#include <fstream> 
#include <stdio.h>  
#include <stdlib.h>
//#include "IniFile.h"
#include <cstring>
#include <string> 
#include <chrono> 
#include <math.h>
#include <time.h>   
#include <iostream>

#include<cuda_runtime.h>
#include <device_launch_parameters.h>

#include "scatter.h"

using namespace std;

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
	
	int numCollimatorLayers = parameter_Collimator[0];
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
	int numProjectionsingle = parameter_Detector[0];

	int numImageVoxelX = parameter_Image[0];
	int numImageVoxelY = parameter_Image[1];
	int numImageVoxelZ = parameter_Image[2];
	float widthImageVoxelX = parameter_Image[3];
	float widthImageVoxelY = parameter_Image[4];
	float widthImageVoxelZ = parameter_Image[5];
	int numRotation_ = (int)floor(parameter_Image[6]+0.001);
	float angelPerRotation = parameter_Image[7];
	float shiftFOVX= parameter_Image[8];
	float shiftFOVY = parameter_Image[9];
	float shiftFOVZ = parameter_Image[10];

	const int numProjectionSingle = numProjectionsingle;
	const int numImagebin = numImageVoxelX * numImageVoxelY * numImageVoxelZ;
	const int numRotation = numRotation_;

	string FnamePE;
	string FnameGeo = "GeometryRelationShip_Crystal2Crystal"; 
	int cuda_id = 0; 

	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-PE") == 0 && i + 1 < argc)
		{
			FnamePE = argv[i + 1];
			i++; 
		}
		else if (strcmp(argv[i], "-GeoCrystal") == 0 && i + 1 < argc)
		{
			FnameGeo = argv[i + 1];
			parameter_Physics[8] = 0;
			i++; 
		}
		else if (strcmp(argv[i], "-cuda") == 0 && i + 1 < argc)
		{
			cuda_id = atoi(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path] [-cuda GPU_ID]" << endl;
			return 0;
		}
		else
		{
			cerr << "Unknown parameter or missing argument: " << argv[i] << endl;
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path]" << endl;
			return EXIT_FAILURE;
		}
	}

	float* PE_SysMat = new float[numProjectionSingle * numImagebin * numRotation]();

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
	fread(PE_SysMat, sizeof(float), numProjectionSingle * numImagebin * numRotation, fp0);
	fclose(fp0);
	auto end_ioPE = std::chrono::high_resolution_clock::now();
	auto duration_ioPE = std::chrono::duration_cast<std::chrono::milliseconds>(end_ioPE - start_ioPE);
	cout << "Time of io PE System Matrix: " << duration_ioPE.count() << " ms" << endl;


	cout << "Geometry RelationShip Crystal2Crystal:  " << FnameGeo << endl;


	auto start_scatter = std::chrono::high_resolution_clock::now();


	float* out = new float[numProjectionSingle * numImagebin * numRotation]();

	printf("FOV dimension : %d %d %d\n", numImageVoxelX, numImageVoxelY, numImageVoxelZ);
	printf("FOV Voxel Size(mm) : %f %f %f\n", widthImageVoxelX, widthImageVoxelY, widthImageVoxelZ);
	for (int idxRotation = 0; idxRotation < numRotation; idxRotation++)
	{
		cout << "########################" << endl;
		cout << "Rotation (" << idxRotation << ") processing ..." << endl;
		cout << "########################" << endl;


		cout << "Shift FOV in X = " << shiftFOVX << "mm" << endl;
		cout << "Shift FOV in Y = " << shiftFOVY << "mm" << endl;
		cout << "Shift FOV in Z = " << shiftFOVZ << "mm" << endl;

		parameter_Image[20] = float(idxRotation);

		int q = scatter(parameter_Detector, parameter_Image, parameter_Physics, PE_SysMat, FnameGeo.c_str(), out + idxRotation * numProjectionSingle * numImagebin, cuda_id);

		printf("numImagebin = %d\n", q);
	}


	auto end_scatter = std::chrono::high_resolution_clock::now();
	auto duration_scatter = std::chrono::duration_cast<std::chrono::milliseconds>(end_scatter - start_scatter);
	cout << "Time of io scatter function: " << duration_scatter.count()/1000.0/60.0 << " min" << endl;


	char Fname[2048];
	sprintf(Fname, "Scatter_Crystal_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
	FILE* fp1;
	fp1 = fopen(Fname, "wb+");
	if (fp1 == 0) { puts("error"); exit(0); }
	fwrite(out, sizeof(float), numProjectionSingle * numImagebin * numRotation, fp1);
	fclose(fp1);

	cout << "########################" << endl;
	cout << "Inter-crystal Compton Scatter Sysmat written." << endl;
	cout << "########################" << endl;

	if (parameter_Physics[3] == 1) 
	{
		float* SysMat = new float[numProjectionSingle * numImagebin * numRotation]();
		for (int i = 0; i < numProjectionSingle * numImagebin * numRotation; i++) 
		{
			SysMat[i] = PE_SysMat[i] + out[i];
		}
		
		char Fname3[2048];
		sprintf(Fname3, "SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FILE* fp2;
		fp2 = fopen(Fname3, "wb+");
		if (fp2 == 0) { puts("error"); exit(0); }
		fwrite(SysMat, sizeof(float), numProjectionSingle * numImagebin * numRotation, fp2);
		fclose(fp2);

		cout << "########################" << endl;
		cout << "Full Sysmat written." << endl;
		cout << "########################" << endl;
	}
	return 0;
}

