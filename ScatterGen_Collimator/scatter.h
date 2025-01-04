#ifndef _PHOTODETECTORCUDA_H_
#define _PHOTODETECTORCUDA_H_

extern int scatter(
	float* parameter_Collimator, float* parameter_Detector, float* parameter_Image, float* parameter_Physics, float* PE_SysMat, const char* FnameGeo, float* dst,int cuda_id);


#endif //_PHOTODETECTORCUDA_H_
