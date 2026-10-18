#define _CRT_SECURE_NO_WARNINGS
 
// PEGen_CircularHole.cpp
// Description:
//   This program calculates the photon-electric system matrix.
//   Fixed for integer overflow with large detector*voxel counts.
//
// Usage:
//   ./PEGen_CircularHole -cuda <cuda_device_id>   (reads Params_*.dat in the working directory)
 
#include <fstream> 
#include <stdio.h>  
#include <stdlib.h>
#include <math.h>
#include <time.h>   
#include <iostream>
#include <cstring>
#include <string> 
 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
 
#include "PESysMatGen.h"
using namespace std;

// Read up to n floats of a parameter file; exits with a message if the file is missing.
static void readParams(const char* name, float* dst, size_t n)
{
	FILE* f = fopen(name, "rb");
	if (!f) { cerr << "error: cannot open " << name << endl; exit(EXIT_FAILURE); }
	size_t got = fread(dst, sizeof(float), n, f);
	fclose(f);
	if (got == 0) { cerr << "error: " << name << " is empty" << endl; exit(EXIT_FAILURE); }
}
 
int main(int argc, char* argv[])
{
	float* parameter_Collimator = new float[80000]();
	float* parameter_Detector = new float[80000]();
	float* parameter_Image = new float[100]();
	float* parameter_Physics = new float[100]();
 
	readParams("Params_Collimator.dat", parameter_Collimator, 80000);
	readParams("Params_Detector.dat", parameter_Detector, 80000);
	readParams("Params_Image.dat", parameter_Image, 100);
	readParams("Params_Physics.dat", parameter_Physics, 100);

	// the parameter files are read into 80000 floats: at most 9 collimator layers, holes up to index 79999,
	// crystals up to index 79999
	{
		int nLayers = (int)floor(parameter_Collimator[0] + 0.001f);
		long long nHoles = 0;
		for (int l = 0; l < nLayers && l < 9; l++) nHoles += (long long)floor(parameter_Collimator[(l + 1) * 10] + 0.001f);
		long long nDet = (long long)floor(parameter_Detector[0] + 0.001f);
		if (nLayers < 0 || nLayers > 9) { cerr << "error: " << nLayers << " collimator layers (at most 9)" << endl; return EXIT_FAILURE; }
		if (100 + 9 * nHoles > 80000) { cerr << "error: " << nHoles << " holes do not fit in Params_Collimator (at most 8877)" << endl; return EXIT_FAILURE; }
		if (12 * nDet + 1 > 80000) { cerr << "error: " << nDet << " crystals do not fit in Params_Detector (at most 6666)" << endl; return EXIT_FAILURE; }
	}
 
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
	for (size_t idxRotation = 0; idxRotation < numRotation; idxRotation++)
	{
		cout << "########################" << endl;
		cout << "Rotation (" << idxRotation << ") processing ..." << endl;
		cout << "########################" << endl;
 
		cout << "Shift FOV in X = " << shiftFOVX << "mm" << endl;
		cout << "Shift FOV in Y = " << shiftFOVY << "mm" << endl;
		cout << "Shift FOV in Z = " << shiftFOVZ << "mm" << endl;
 
		parameter_Image[20] = float(idxRotation);
 
		int q = PESysMatGen(parameter_Collimator, parameter_Detector, parameter_Image, 
		                     out + idxRotation * numProjectionSingle * numImagebin, cuda_id);
 
		if (q < 0) { cerr << "error: PE system matrix calculation failed (rotation " << idxRotation << "), nothing written" << endl; delete[] out; return EXIT_FAILURE; }
		printf("numImagebin = %d\n", q);
	}
 
	char Fname[2048];
	sprintf(Fname, "PE_SysMat_shift_%f_%f_%f_v3.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
	FILE* fp1;
	fp1 = fopen(Fname, "wb+");
	if (fp1 == 0) { cerr << "error: cannot write " << Fname << endl; delete[] out; return EXIT_FAILURE; }
	size_t written = fwrite(out, sizeof(float), totalElements, fp1);
	fclose(fp1);
	if (written != totalElements) { cerr << "error: writing " << Fname << " failed (disk full?)" << endl; delete[] out; return EXIT_FAILURE; }
 
	delete[] out;
 
	cout << "########################" << endl;
	cout << "Photon Electric Sysmat Written." << endl;
	cout << "########################" << endl;
	return 0;
}
 
