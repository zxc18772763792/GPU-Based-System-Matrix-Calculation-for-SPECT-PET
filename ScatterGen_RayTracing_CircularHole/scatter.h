#ifndef _PHOTODETECTORCUDA_H_
#define _PHOTODETECTORCUDA_H_

#include <stddef.h>

extern int scatter(
	float* parameter_Collimator, float* parameter_Detector, float* parameter_Image, float* parameter_Physics, float* PE_SysMat, const char* FnameGeoCrystal, const char* FnameGeoCollimator, float* dst,int cuda_id,
	size_t imageBinStart = 0, size_t imageBinCount = 0);


#endif //_PHOTODETECTORCUDA_H_
