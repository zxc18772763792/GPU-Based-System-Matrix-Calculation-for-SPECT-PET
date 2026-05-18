#pragma once
#ifndef PESYSMATGEN_H
#define PESYSMATGEN_H
 
#include <stddef.h>

int PESysMatGen(float* parameter_Collimator, float* parameter_Detector,
                float* parameter_Image, float* dst, int cuda_id,
                size_t imageBinStart = 0, size_t imageBinCount = 0);
 
#endif
