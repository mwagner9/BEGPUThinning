/**
 * @file Matlab3DThinning.cpp
 *
 * @brief This files implements the matlab interface for the centerline extraction.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */


/*==========================================================
* Matlab3DThinning - MATLAB External Interface for 3D Thinning
*
* Extracts the 3D Centerlines (outMatrix) 
* of a binary volume (inMatrix)
*
* The calling syntax is:
*
*		outMatrix = Matlab3DThinning(inMatrix)
*
* This is a MEX-file for MATLAB.
* Copyright 2007-2012 The MathWorks, Inc.
*
*========================================================*/

#include "mex.h"
#include "LibThinning3D.h"
#include "ExtractSegments.h"
#include "CenterlineExctraction.h"

#include <algorithm>

#define USE_SEPARATE_COMPLEX_API  // Comment out to use interleaved API (currently not supported by mexcuda as of R2018b)

// The gateway function 
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{

	// check for proper number of arguments 
	if (nrhs < 1) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "One inputs required.");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "One output required.");
	}

	// check that first input argument is a valid 3D array
	mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
	const mwSize * dims = mxGetDimensions(prhs[0]);;

	if ((ndims != 3) || (dims[0] <= 1) || (dims[1] <= 1) || (dims[2] <= 1)) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:not3DMatrix", "Input must be 3D matrix.");
	}

	// call the computational routine
	float *ox = 0, *oy = 0, *oz = 0;
	unsigned int *oi = 0;
	int nsegs;

	// Get parameters
	int pruning_length = 0;
	int smoothing_span = 0;

	if (nrhs > 1) pruning_length = std::max<int>(0, int(mxGetScalar(prhs[1])));
	if (nrhs > 2) smoothing_span = std::max<int>(0, int(mxGetScalar(prhs[2])));

	// Check input array type
	if (mxIsDouble(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		double *indata = mxGetPr(prhs[0]); 
#else
		double *indata = mxGetDoubles(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<double>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsSingle(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		float *indata = (float *)mxGetData(prhs[0]);
#else
		float *indata = mxGetSingles(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<float>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsInt32(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		int *indata = (int *)mxGetData(prhs[0]);
#else
		int *indata = mxGetInt32s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<int>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsInt16(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		short *indata = (short *)mxGetData(prhs[0]);
#else
		short *indata = mxGetInt16s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<short>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsInt8(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		signed char *indata = (signed char *)mxGetData(prhs[0]);
#else
		signed char *indata = mxGetInt8s(prhs[0]); 
#endif
		nsegs = CenterlineExtraction<signed char>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsUint32(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		unsigned int *indata = (unsigned int *)mxGetData(prhs[0]);
#else
		unsigned int *indata = mxGetUint32s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<unsigned int>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsUint16(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		unsigned short *indata = (unsigned short *)mxGetData(prhs[0]);
#else
		unsigned short *indata = mxGetUint16s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<unsigned short>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsUint8(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		unsigned char *indata = (unsigned char *)mxGetData(prhs[0]);
#else
		unsigned char *indata = mxGetUint8s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<unsigned char>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else if (mxIsLogical(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		bool *indata = (bool *)mxGetData(prhs[0]);
#else
		bool *indata = mxGetLogicals(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<bool>(indata, int(dims[0]), int(dims[1]), int(dims[2]), pruning_length, smoothing_span, &ox, &oy, &oz, &oi);
	}
	else
	{
		mexErrMsgIdAndTxt("Matlab3DThinning:input:prhs", "Unsupport data type.");
	}

	// extract segments
	plhs[0] = mxCreateCellMatrix(nsegs, 1);
	for (int i = 0; i < nsegs; i++)
	{		
		int ssize = oi[i+1] - oi[i];
		mxArray *s = mxCreateDoubleMatrix((mwSize)ssize, 3, mxREAL);
		double *ds = mxGetPr(s);
		for (int pi = oi[i], ti = 0; pi < oi[i+1]; pi++, ti++)
		{
			ds[ti] = (double)oy[pi];
			ds[ssize + ti] = (double)ox[pi];
			ds[2 * ssize + ti] = (double)oz[pi];
		}
		mxSetCell(plhs[0], i, s);		
	}

	// Clean up
	if (ox) delete[] ox;
	if (oy) delete[] oy;
	if (oz) delete[] oz;
	if (oi) delete[] oi;
}