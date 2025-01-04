#define _CRT_SECURE_NO_WARNINGS

// PEGen_CircularHole.cpp
// Description:
//   This program calculates the photon-electric system matrix.
//
// Usage:
//   ./PE_SysMat_Gen -cuda <cuda_device_id>
//
// Author: Xingchun Zheng
// Last Modified: 2024/12/22
// Version: 1.0

#include <fstream> 
#include <stdio.h>  
#include <stdlib.h>


#include <math.h>
#include <time.h>   
#include <iostream>
#include <cstring>
#include <string> 

#include<cuda_runtime.h>
#include <device_launch_parameters.h>

#include "PESysMatGen.h"
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
	int cuda_id = 0; 

	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "-cuda") == 0 && i + 1 < argc)
		{
			cuda_id = atoi(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << "Usage: " << argv[0] << " [-cuda GPU_ID]" << endl;
			return 0;
		}
		else
		{
			cerr << "Unknown parameter or missing argument: " << argv[i] << endl;
			cout << "Usage: " << argv[0] << " [-cuda GPU_ID] " << endl;
			return EXIT_FAILURE;
		}
	}

	////////////////////////////////////////////////////
	int numProjectionsingle = (int)floor(parameter_Detector[0]+0.001f);

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

	const int numProjectionSingle = numProjectionsingle;
	const int numImagebin = numImageVoxelX * numImageVoxelY * numImageVoxelZ;
	const int numRotation = numRotation_;

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

		int q = PESysMatGen(parameter_Collimator, parameter_Detector, parameter_Image, out + idxRotation * numProjectionSingle * numImagebin, cuda_id);

		printf("numImagebin = %d\n", q);
		
	}

	char Fname[2048];
	sprintf(Fname, "PE_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
	FILE* fp1;
	fp1 = fopen(Fname, "wb+");
	if (fp1 == 0) { puts("error"); exit(0); }
	fwrite(out, sizeof(float), numProjectionSingle * numImagebin * numRotation, fp1);
	fclose(fp1);


	cout << "########################" << endl;
	cout << "Photon Electric Sysmat Written." << endl;
	cout << "########################" << endl;
	//system("Pause");
	return 0;
}

