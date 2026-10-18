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
//                -components   (also write Scatter_Crystal_SysMat_... and Scatter_Collimator_SysMat_...)
//                -model v2|legacy   (default v2: volume-integrated single-scatter model, see scatter.cu)
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

#include<cuda_runtime.h>
#include <device_launch_parameters.h>

#include "scatter.h"

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

// Write n floats; exits with a message on failure.
static void writeMatrix(const char* name, const float* src, size_t n)
{
	FILE* f = fopen(name, "wb+");
	if (!f) { cerr << "error: cannot create " << name << endl; exit(EXIT_FAILURE); }
	size_t put = fwrite(src, sizeof(float), n, f);
	if (fclose(f) != 0 || put != n) { cerr << "error: writing " << name << " failed" << endl; exit(EXIT_FAILURE); }
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
	
	int numCollimatorLayers = (int)floor(parameter_Collimator[0] + 0.001f);
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
	int numProjectionsingle = (int)floor(parameter_Detector[0] + 0.001f);

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
	// sizes in size_t: crystals x voxels x rotations easily exceeds 2^31
	const size_t perRotation = (size_t)numProjectionSingle * numImagebin;
	const size_t numElements = perRotation * numRotation;
	if (numProjectionSingle < 1 || numImagebin < 1 || numRotation < 1)
	{
		cerr << "error: " << numProjectionSingle << " crystals, " << numImagebin << " voxels, " << numRotation << " rotations" << endl;
		return EXIT_FAILURE;
	}

	string FnamePE;
	string FnameGeoCrystal = "GeometryRelationShip_Crystal2Crystal"; 
	string FnameGeoCollimator = "GeometryRelationShip_Collimator2Crystal";
	int cuda_id = 0; 
	bool writeComponents = false;
	int model = 2;

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
		else if (strcmp(argv[i], "-components") == 0)
		{
			writeComponents = true;
		}
		else if (strcmp(argv[i], "-model") == 0 && i + 1 < argc)
		{
			if (strcmp(argv[i + 1], "legacy") == 0) model = 1;
			else if (strcmp(argv[i + 1], "v2") == 0) model = 2;
			else { cerr << "-model must be v2 or legacy" << endl; return EXIT_FAILURE; }
			i++;
		}
		else if (strcmp(argv[i], "-cuda") == 0 && i + 1 < argc)
		{
			cuda_id = atoi(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path] [-GeoCollimator GeometryRelationShip_Collimator2Crystal_path] [-cuda GPU_ID] [-components] [-model v2|legacy]" << endl;
			return 0;
		}
		else
		{
			cerr << "Unknown parameter or missing argument: " << argv[i] << endl;
			cout << "Usage: " << argv[0] << " [-PE PE_SysMat_path] [-GeoCrystal GeometryRelationShip_Crystal2Crystal_path] [-GeoCollimator GeometryRelationShip_Collimator2Crystal_path] " << endl;
			return EXIT_FAILURE;
		}
	}

	//////////////////////////// PE SysMat Loading ///////////////////////////////
	float* PE_SysMat = new float[numElements]();

	if (FnamePE.empty())
	{
		// default: the name PEGen_CircularHole writes (..._v3.sysmat), else the older name without _v3
		char bufferPE[2048];
		snprintf(bufferPE, sizeof(bufferPE), "PE_SysMat_shift_%f_%f_%f_v3.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		FnamePE = bufferPE;
		FILE* probe = fopen(bufferPE, "rb");
		if (probe) fclose(probe);
		else
		{
			snprintf(bufferPE, sizeof(bufferPE), "PE_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
			probe = fopen(bufferPE, "rb");
			if (probe) { fclose(probe); FnamePE = bufferPE; }
		}
	}

	cout << "Photon Electric SysMat: " << FnamePE << endl;

	auto start_ioPE = std::chrono::high_resolution_clock::now();
	FILE* fp0;
	fp0 = fopen(FnamePE.c_str(), "rb");
	if (fp0 == 0) { cerr << "error: cannot open the PE system matrix " << FnamePE << endl; return EXIT_FAILURE; }
	size_t gotPE = fread(PE_SysMat, sizeof(float), numElements, fp0);
	fclose(fp0);
	if (gotPE != numElements)
	{
		cerr << "error: " << FnamePE << " holds " << gotPE << " floats, expected " << numElements
			<< " (" << numProjectionSingle << " crystals x " << numImagebin << " voxels x " << numRotation << " rotations)" << endl;
		return EXIT_FAILURE;
	}
	auto end_ioPE = std::chrono::high_resolution_clock::now();
	auto duration_ioPE = std::chrono::duration_cast<std::chrono::milliseconds>(end_ioPE - start_ioPE);
	cout << "Time of io PE System Matrix: " << duration_ioPE.count() << " ms" << endl;

	//////////////////////////// Scatter Function Start ///////////////////////////////
	cout << "Geometry RelationShip Crystal2Crystal:  " << FnameGeoCrystal << endl;
	cout << "Geometry RelationShip Collimator2Crystal:  " << FnameGeoCollimator << endl;
	cout << "Scatter model: " << (model == 2 ? "v2 (volume-integrated single scatter)" : "legacy") << endl;

	auto start_scatter = std::chrono::high_resolution_clock::now();

	float* out = new float[numElements]();
	float* outCrystal = writeComponents ? new float[numElements]() : 0;
	float* outCollimator = writeComponents ? new float[numElements]() : 0;

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

		const size_t off = idxRotation * perRotation;
		int q = scatter(parameter_Collimator, parameter_Detector, parameter_Image, parameter_Physics, PE_SysMat + off, FnameGeoCrystal.c_str(), FnameGeoCollimator.c_str(), out + off, cuda_id,
			writeComponents ? outCrystal + off : 0,
			writeComponents ? outCollimator + off : 0, model);
		if (q < 0) { cerr << "error: scatter() failed for rotation " << idxRotation << ", nothing written" << endl; return EXIT_FAILURE; }

		printf("numImagebin = %d\n", q);

		// the geometry bitmaps do not depend on the rotation: computed (and written) in the first rotation,
		// read back from that file in the following ones
		parameter_Physics[8] = 0;
		parameter_Physics[9] = 0;
	}

	auto end_scatter = std::chrono::high_resolution_clock::now();
	auto duration_scatter = std::chrono::duration_cast<std::chrono::milliseconds>(end_scatter - start_scatter);
	cout << "Time of scatter function: " << duration_scatter.count()/1000.0/60.0 << " min" << endl;

	if (parameter_Physics[2] == 1)
	{
		char Fname[2048];
		sprintf(Fname, "Scatter_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		writeMatrix(Fname, out, numElements);

		cout << "########################" << endl;
		cout << "Compton Scatter Sysmat written." << endl;
		cout << "########################" << endl;
	}
	if (writeComponents)
	{
		const char* names[2] = { "Scatter_Crystal_SysMat_shift_%f_%f_%f.sysmat", "Scatter_Collimator_SysMat_shift_%f_%f_%f.sysmat" };
		float* parts[2] = { outCrystal, outCollimator };
		for (int k = 0; k < 2; k++)
		{
			char Fname[2048];
			sprintf(Fname, names[k], shiftFOVX, shiftFOVY, shiftFOVZ);
			writeMatrix(Fname, parts[k], numElements);
			cout << "Component written: " << Fname << endl;
		}
	}
	if (parameter_Physics[3] == 1) 
	{
		float* SysMat = new float[numElements]();
		for (size_t i = 0; i < numElements; i++) 
		{
			SysMat[i] = PE_SysMat[i] + out[i];
		}
		
		char Fname3[2048];
		sprintf(Fname3, "SysMat_withScatter_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		writeMatrix(Fname3, SysMat, numElements);

		cout << "########################" << endl;
		cout << "Full Sysmat written." << endl;
		cout << "########################" << endl;
	}
	return 0;
}

