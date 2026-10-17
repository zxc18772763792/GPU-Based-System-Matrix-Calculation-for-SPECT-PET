#define _CRT_SECURE_NO_WARNINGS

// ScatterGen_Crystal.cpp  (legacy kernels; the maintained module is ScatterGen_RayTracing_CircularHole)
// Calculate the scatter system matrix, with primary compton events only
// run this program after the PE system matrix generation
// usage: ./ScatterGen_Crystal -PE "path/to/PE_SystemMatrix" -GeoCrystal "path/to/CrystalGeometryRelationship" -cuda int
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
	// sizes in size_t: crystals x voxels x rotations easily exceeds 2^31
	const size_t perRotation = (size_t)numProjectionSingle * numImagebin;
	const size_t numElements = perRotation * numRotation;
	if (numProjectionSingle < 1 || numImagebin < 1 || numRotation < 1)
	{
		cerr << "error: " << numProjectionSingle << " crystals, " << numImagebin << " voxels, " << numRotation << " rotations" << endl;
		return EXIT_FAILURE;
	}

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


	cout << "Geometry RelationShip Crystal2Crystal:  " << FnameGeo << endl;


	auto start_scatter = std::chrono::high_resolution_clock::now();


	float* out = new float[numElements]();

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

		int q = scatter(parameter_Detector, parameter_Image, parameter_Physics, PE_SysMat + idxRotation * perRotation, FnameGeo.c_str(), out + idxRotation * perRotation, cuda_id);
		if (q < 0) { cerr << "error: scatter() failed for rotation " << idxRotation << ", nothing written" << endl; return EXIT_FAILURE; }

		printf("numImagebin = %d\n", q);

		// the geometry bitmaps do not depend on the rotation: computed (and written) in the first rotation,
		// read back from that file in the following ones
		parameter_Physics[8] = 0;
		parameter_Physics[9] = 0;
	}


	auto end_scatter = std::chrono::high_resolution_clock::now();
	auto duration_scatter = std::chrono::duration_cast<std::chrono::milliseconds>(end_scatter - start_scatter);
	cout << "Time of io scatter function: " << duration_scatter.count()/1000.0/60.0 << " min" << endl;


	// "_legacy" names: the maintained module ScatterGen_RayTracing_CircularHole writes
	// Scatter_SysMat_*, Scatter_Crystal_SysMat_* and Scatter_Collimator_SysMat_* (model v2)
	char Fname[2048];
	sprintf(Fname, "Scatter_Crystal_legacy_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
	writeMatrix(Fname, out, numElements);
	cout << "written: " << Fname << endl;

	cout << "########################" << endl;
	cout << "Inter-crystal Compton Scatter Sysmat written." << endl;
	cout << "########################" << endl;

	if (parameter_Physics[3] == 1) 
	{
		float* SysMat = new float[numElements]();
		for (size_t i = 0; i < numElements; i++) 
		{
			SysMat[i] = PE_SysMat[i] + out[i];
		}
		
		char Fname3[2048];
		sprintf(Fname3, "SysMat_withCrystalScatter_legacy_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);
		writeMatrix(Fname3, SysMat, numElements);
		cout << "written: " << Fname3 << endl;

		cout << "########################" << endl;
		cout << "Full Sysmat written." << endl;
		cout << "########################" << endl;
	}
	return 0;
}

