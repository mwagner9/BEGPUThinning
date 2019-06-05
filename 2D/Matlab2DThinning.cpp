/**
 * @file Matlab2DThinning.cpp
 *
 * @brief This files implements the matlab interface for the centerline extraction.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

/*==========================================================
* Matlab2DThinning - MATLAB External Interface for 2D Thinning
*
* Extracts the 2D Centerlines (outMatrix)
* of a binary volume (inMatrix)
*
* The calling syntax is:
*
*		outMatrix = Matlab2DThinning(inMatrix, pruning_length = 0, smoothing_span = 1)
*
*
*========================================================*/

#include "mex.h"
#include "matrix.h"
#include "LibThinning2D.h"
#include "ExtractSegments.h"
#include "CenterlineExtraction.h"

#include <algorithm>

#define USE_SEPARATE_COMPLEX_API  // Comment out to use interleaved API (currently not supported by mexcuda as of R2018b)


// The gateway function 
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{

	// check for proper number of arguments 
	if (nrhs < 1) {
		mexErrMsgIdAndTxt("Matlab2DThinning:input:nrhs", "At least one input required.");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("Matlab2DThinning:input:nlhs", "One output required.");
	}

	// Check if array
	if (mxGetNumberOfDimensions(prhs[0]) != 2)
	{
		mexErrMsgIdAndTxt("Matlab2DThinning:input:prhs", "Input has to be a 2D array.");
	}

	// Check if array is real
	if (mxIsComplex(prhs[0]))
	{
		mexErrMsgIdAndTxt("Matlab2DThinning:input:prhs", "Input array cannot be complex.");
	}

	// Get parameters
	int pruning_length = 0;
	int smoothing_span = 0;

	if (nrhs > 1) pruning_length = std::max<int>(0, int(mxGetScalar(prhs[1])));
	if (nrhs > 2) smoothing_span = std::max<int>(0, int(mxGetScalar(prhs[2])));

	// Get array size
	const mwSize *sz = mxGetDimensions(prhs[0]);
		
	// Initialize variables
	unsigned int nsegs;
	float *ox = 0, *oy = 0;
	unsigned int *oi = 0;

	// Check input array type
	if (mxIsDouble(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		double *indata = mxGetPr(prhs[0]); 
#else
		double *indata = mxGetDoubles(prhs[0]); 
#endif
		nsegs = CenterlineExtraction<double>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	} 
	else if (mxIsSingle(prhs[0]))
	{ 
#ifdef USE_SEPARATE_COMPLEX_API
		float *indata = (float *)mxGetData(prhs[0]);
#else
		float *indata = mxGetSingles(prhs[0]); 
#endif
		nsegs = CenterlineExtraction<float>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else if (mxIsInt32(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		int *indata = (int *)mxGetData(prhs[0]);
#else
		int *indata = mxGetInt32s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<int>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else if (mxIsInt16(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		short *indata = (short *)mxGetData(prhs[0]); 
#else
		short *indata = mxGetInt16s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<short>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else if (mxIsInt8(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		signed char *indata = (signed char *)mxGetData(prhs[0]); 
#else
		signed char *indata = mxGetInt8s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<signed char>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else if (mxIsUint32(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		unsigned int *indata = (unsigned int *)mxGetData(prhs[0]); 
#else
		unsigned int *indata = mxGetUint32s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<unsigned int>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else if (mxIsUint16(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		unsigned short *indata = (unsigned short *)mxGetData(prhs[0]); 
#else
		unsigned short *indata = mxGetUint16s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<unsigned short>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else if (mxIsUint8(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		unsigned char *indata = (unsigned char *) mxGetData(prhs[0]);
#else
		unsigned char *indata = mxGetUint8s(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<unsigned char>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else if (mxIsLogical(prhs[0]))
	{
#ifdef USE_SEPARATE_COMPLEX_API
		bool *indata = (bool *) mxGetData(prhs[0]); 
#else
		bool *indata = mxGetLogicals(prhs[0]); 
#endif		
		nsegs = CenterlineExtraction<bool>(indata, int(sz[0]), int(sz[1]), pruning_length, smoothing_span, &ox, &oy, &oi);
	}
	else
	{
		mexErrMsgIdAndTxt("Matlab2DThinning:input:prhs", "Unsupport data type.");
	}

	// extract segments
	plhs[0] = mxCreateCellMatrix(nsegs, 1);
	for (int i = 0; i < int(nsegs); i++)
	{
		int ssize = oi[i + 1] - oi[i];
		mxArray *s = mxCreateDoubleMatrix((mwSize)ssize, 2, mxREAL);
		double *ds = mxGetPr(s);
		for (int pi = int(oi[i]), ti = 0; pi < int(oi[i + 1]); pi++, ti++)
		{
			ds[ti] = (double)oy[pi];
			ds[ssize + ti] = (double)ox[pi];
		}
		mxSetCell(plhs[0], i, s);
	}

	// Clean up
	if (ox) delete[] ox;
	if (oy) delete[] oy;
	if (oi) delete[] oi;
}