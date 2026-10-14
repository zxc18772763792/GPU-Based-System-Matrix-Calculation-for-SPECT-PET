#ifndef _PHOTODETECTORCUDA_H_
#define _PHOTODETECTORCUDA_H_

extern int scatter(
	float* parameter_Collimator, float* parameter_Detector, float* parameter_Image, float* parameter_Physics, float* PE_SysMat, const char* FnameGeoCrystal, const char* FnameGeoCollimator, float* dst,int cuda_id,
	float* dstCrystal = 0, float* dstCollimator = 0); // optional: the two parts of dst


#endif //_PHOTODETECTORCUDA_H_
