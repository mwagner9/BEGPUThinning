/**
 * @file LibThinning2D.h
 *
 * @brief This files implements the thinning function for binary images.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#include "LibThinning2D.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "CUDA_DeviceHelperFunctions.cuh"
#include "CUDA_ThinningKernelMultiBlock2D.cuh"

//#define SAFE_MODE 1  // Uncomment this for debug purposes to perform error checks for all CUDA calls
#define	WARP_SIZE 32


#ifdef SAFE_MODE
	#define CUDA_SAFE_CALL(x) {				\
			if (x != cudaSuccess)			\
							{								\
				printf("Cuda error %i occured in %s, line %i!\n", x, __FILE__, __LINE__); \
				std::cin.ignore(1);		\
				exit(-1);				\
							} \
									}
#else
	#define CUDA_SAFE_CALL(x) x
#endif


template< typename PixelType>
void Thinning2D(int width, int height, PixelType *data, PixelType *out)
{

	// has to be padded to height = (x+0.5)*overlapRows, where x is an arbitrary integer value and width is a multiple of 32. 
	//
	//	Parameters:
	//		gdata		-		Binary input image. Each pixel is stored as a single bit.
	//		odata		-		Thinned output image. Each pixel is stored as single bit.
	//		overlapRows	-		Number of overlapping rows in adjacent CUDA blocks
	//
	//	Template Parameters:
	//		NBlockY		-		Number of rows that are processed by one set of CUDA threads
	//
	//	Shared Memory Size: NBlockY*blockDim.y*blockDim.x*sizeof(unsigned int) + 32 * sizeof(unsigned int)
	//  Number of Cuda Threads: dim3(<Num ints per row>, floor(1024.0 / blockDim.x))
	//	Number of Cuda Blocks: dim3(1,(ImageHeight - overlapRows) / (blockDim.y*NBlockY - overlapRows))
	// ----------------------------------------------------------------------------------------------------------------------------

	// Pad image to multiple of 32 pixels and power of 2 in x-dimension and height to (a*ny*by+(a-1)*overlapRows)
	const int overlap = 64;
	const int ny = 8;
	int sx = 32;
	while (sx < width) sx <<= 1;
	int bx = sx / 32;
	int by = 1024 / bx;
	int b = (ny*by - overlap);
	int a = int(((height - overlap) + b - 1) / b);
	int sy = a*ny*by - (a - 1)*overlap;
	size_t numelem = sx*sy;

	// Allocate Temporary Device Memory
	PixelType *dInputOutput;
	unsigned int *dEncoded1, *dEncoded2;
	CUDA_SAFE_CALL(cudaMalloc(&dInputOutput, sizeof(PixelType) * numelem));
	CUDA_SAFE_CALL(cudaMalloc(&dEncoded1, sizeof(unsigned int) * (numelem / 32))); 
	CUDA_SAFE_CALL(cudaMalloc(&dEncoded2, sizeof(unsigned int) * (numelem / 32)));

	// Copy Data To GPU
	CUDA_SAFE_CALL(cudaMemset(dInputOutput, 0, numelem*sizeof(PixelType)));
	CUDA_SAFE_CALL(cudaMemcpy2D(dInputOutput, sx*sizeof(PixelType), data, width*sizeof(PixelType), width*sizeof(PixelType), height, cudaMemcpyHostToDevice));

	// Encode Image
	int ws = WARP_SIZE*WARP_SIZE*WARP_SIZE;
	EncodeBitsKernel<PixelType> << < ((sx * sy + ws - 1) / ws), dim3(WARP_SIZE, WARP_SIZE) >> >(dInputOutput, dEncoded1, sx, sy);
#ifdef SAFE_MODE
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif

	// Calculate Number of Blocks
	dim3 grid(1, (sy - overlap) / (ny*by - overlap));
	dim3 block(bx, by);
	unsigned int smemsize = ny*by*bx*sizeof(unsigned int) + 32 * sizeof(unsigned int);

	int nblocks = grid.y;
	unsigned int *dcount, *hcount = new unsigned int[nblocks];
	CUDA_SAFE_CALL(cudaMalloc(&dcount, sizeof(int)*nblocks));

	// Thinning
	int count = 1;
	unsigned int *ti = dEncoded1;
	unsigned int *ri = dEncoded2;

	while (count > 0)
	{
		ThinningKernelMultiBlock2D2<ny> << < grid, block, smemsize >> >(ti, ri, overlap, dcount);
		cudaMemcpy(hcount, dcount, sizeof(unsigned int)*nblocks, cudaMemcpyDeviceToHost);
		count = 0;
		for (int i = 0; i < nblocks; i++) count += hcount[i];

		unsigned int *swapimg = ti;
		ti = ri;
		ri = swapimg;
	}

	// Decode Image
	ExtractBitsKernel<PixelType> << < ((sx * sy + ws - 1) / ws), dim3(WARP_SIZE, WARP_SIZE) >> >(ti, dInputOutput, sx, sy);
#ifdef SAFE_MODE
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif

	// Copy Output to CPU
	CUDA_SAFE_CALL(cudaMemcpy2D(out, width*sizeof(PixelType), dInputOutput, sx*sizeof(PixelType), width*sizeof(PixelType), height, cudaMemcpyDeviceToHost));

	// Cleanup Memory
	delete[] hcount;
	CUDA_SAFE_CALL(cudaFree(dcount));
	CUDA_SAFE_CALL(cudaFree(dEncoded1));
	CUDA_SAFE_CALL(cudaFree(dEncoded2));
	CUDA_SAFE_CALL(cudaFree(dInputOutput));

}


// Explicit template instantiation
template void Thinning2D<unsigned char>(int width, int height, unsigned char *data, unsigned char *out);
template void Thinning2D<unsigned short>(int width, int height, unsigned short *data, unsigned short *out);
template void Thinning2D<unsigned int>(int width, int height, unsigned int *data, unsigned int *out);
template void Thinning2D<char>(int width, int height, char *data, char *out);
template void Thinning2D<signed char>(int width, int height, signed char *data, signed char *out);
template void Thinning2D<short>(int width, int height, short *data, short *out);
template void Thinning2D<int>(int width, int height, int *data, int *out);
template void Thinning2D<float>(int width, int height, float *data, float *out);
template void Thinning2D<double>(int width, int height, double *data, double *out);
template void Thinning2D<bool>(int width, int height, bool *data, bool *out);

