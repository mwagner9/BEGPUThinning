/**
 * @file LibThinning3D.h
 *
 * @brief This files implements the thinning function for binary volumes.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#include "LibThinning3D.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "CUDA_DeviceHelperFunctions.cuh"
#include "CUDA_ThinningKernel3D_Linear.cuh"

#define CUDA_SAFE_CALL(x) {				\
		if (x != cudaSuccess)			\
				{								\
			printf("Cuda error %i occured in %s, line %i!\n", x, __FILE__, __LINE__); \
			char buffer[256];		\
			std::cin >> buffer;		\
			exit(-1);				\
				} \
				}
																

#define	WARP_SIZE 32

template< typename PixelType>
void Thinning3D(int width, int height, int depth, PixelType *data, PixelType *out)
{
	size_t size = width*height*depth;

	// Allocate Temporary Device Memory
	PixelType *dInputOutput;
	unsigned short *dEncoded1, *dEncoded2;
	CUDA_SAFE_CALL(cudaMalloc(&dInputOutput, sizeof(PixelType) * size));
	CUDA_SAFE_CALL(cudaMalloc(&dEncoded1, sizeof(unsigned short) * size));
	CUDA_SAFE_CALL(cudaMalloc(&dEncoded2, sizeof(unsigned short) * size));

	// Copy Data To GPU
	CUDA_SAFE_CALL(cudaMemcpy(dInputOutput, data, sizeof(PixelType)*size, cudaMemcpyHostToDevice));

	// Calculate Number of Blocks
	dim3 blocks((width / 32 - 1), (height / 32 - 1), (depth / 32 - 1));
	int nblocks = blocks.x*blocks.y*blocks.z;
	int *dcount, *hcount = new int[nblocks];
	CUDA_SAFE_CALL(cudaMalloc(&dcount, sizeof(int)*nblocks));

	// Encode Image
	EncodeBitsKernelY<PixelType, unsigned short> << < dim3(width / WARP_SIZE, (height*depth) / (8 * sizeof(unsigned short)*WARP_SIZE)), dim3(WARP_SIZE, WARP_SIZE) >> >(dInputOutput, dEncoded1, width, height * depth);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// Thinning
	int count = 1;
	unsigned short *ti = dEncoded1;
	unsigned short *ri = dEncoded2;

	while (count > 0)
	{
		ThinningKernel3DLinearBlock << <blocks, dim3(32, 1, 32) >> >(ti, ri, dcount);
		cudaMemcpy(hcount, dcount, sizeof(unsigned int)*nblocks, cudaMemcpyDeviceToHost);
		count = 0;
		for (int i = 0; i < nblocks; i++) count += hcount[i];

		unsigned short *swapimg = ti;
		ti = ri;
		ri = swapimg;
	}

	// Decode Image
	ExtractBitsKernelY<PixelType, unsigned short> << <  dim3(width / WARP_SIZE, (height*depth) / (8 * sizeof(unsigned short)*WARP_SIZE)), dim3(WARP_SIZE, WARP_SIZE) >> >(ti, dInputOutput, width, height * depth);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// Copy Output to CPU
	CUDA_SAFE_CALL(cudaMemcpy(out, dInputOutput, sizeof(unsigned char)*size, cudaMemcpyDeviceToHost));

	// Cleanup Memory
	delete[] hcount;
	CUDA_SAFE_CALL(cudaFree(dcount));
	CUDA_SAFE_CALL(cudaFree(dEncoded1));
	CUDA_SAFE_CALL(cudaFree(dEncoded2));
	CUDA_SAFE_CALL(cudaFree(dInputOutput));
	
}

template void Thinning3D<unsigned char>(int width, int height, int depth, unsigned char *data, unsigned char *out);
template void Thinning3D<unsigned short>(int width, int height, int depth, unsigned short *data, unsigned short *out);
template void Thinning3D<unsigned int>(int width, int height, int depth, unsigned int *data, unsigned int *out);
template void Thinning3D<char>(int width, int height, int depth, char *data, char *out);
template void Thinning3D<short>(int width, int height, int depth, short *data, short *out);
template void Thinning3D<int>(int width, int height, int depth, int *data, int *out);
template void Thinning3D<float>(int width, int height, int depth, float *data, float *out);
template void Thinning3D<bool>(int width, int height, int depth, bool *data, bool *out);
