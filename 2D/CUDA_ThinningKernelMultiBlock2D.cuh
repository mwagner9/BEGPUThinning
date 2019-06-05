/**
 * @file CUDA_ThinningKernelMultiBlock2D.cuh
 *
 * @brief This files implements the main CUDA kernels to perform 2D thinning.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#ifndef CUDA_THINNING_KERNEL_MULTI_BLOCK_2D_CUH
#define CUDA_THINNING_KERNEL_MULTI_BLOCK_2D_CUH


#ifdef __CUDACC__
#define L(x) __launch_bounds__(x)
#else
#define L(x)
#endif


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA_DeviceHelperFunctions.cuh"
#include "CUDA_SubiterationKernels2D.cuh"

__inline__ __device__ void ReadUpDownDataMB(volatile unsigned int *data, int ty, unsigned int &up, unsigned int &down)
{
	up = data[max(0, ty - 1)*blockDim.x + threadIdx.x];	// x0
	down = data[min(511, ty + 1) * blockDim.x + threadIdx.x]; // x4
}

template< unsigned int NBlockY >
__inline__ __device__ void WriteNewDataMB(volatile unsigned int *data, unsigned int *val)
{
	__syncthreads();
#pragma unroll
	for (int iy = threadIdx.y, idx = 0; idx < NBlockY; iy += blockDim.y, idx++)
	{
		data[iy * blockDim.x + threadIdx.x] = val[idx];
	}
	__syncthreads();
}

// ---------------------------------------------------------------------------------------------------------------------------
// THINNINGKERNELMULTIBLOCK2D2 - Performs 2D thinning of a binary image stored as bits in GDATA using boolean algebra. The 
// whole image width is performed within a single CUDA block. The y-dimension is split up into multiple CUDA blocks. An 
// overlap is required (y-dimension) which defines the number of overlapping rows in adjacent CUDA blocks.  Image size 
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
// Version 1.0 (03/10/2017)
// Author: Martin G. Wagner
// Email: mwagner9@wisc.edu
// ----------------------------------------------------------------------------------------------------------------------------
template < unsigned int NBlockY >
__global__ void L(1024) ThinningKernelMultiBlock2D2(unsigned int *gdata, unsigned int *odata, unsigned int overlapRows, unsigned int *numChanges)
{

	// Initialize Arrays
	extern __shared__ volatile unsigned int sdata[];
	volatile unsigned int *data = sdata;
	volatile int *cmem = (int *)&sdata[NBlockY*blockDim.y*blockDim.x];

	// Calculate Warp Indices
	int lidx = (threadIdx.y * blockDim.x + threadIdx.x);
	int wy = lidx >> 5;
	int wl = (blockDim.x * blockDim.y + 31) >> 5;
	int widx = (wy << 5) ^ lidx;
	unsigned int boffset = (NBlockY*blockDim.y - overlapRows) * blockIdx.y;

	// Load ny Blocks of Data
	unsigned int val[NBlockY];
#pragma unroll
	for (int iy = threadIdx.y, idx = 0; idx < NBlockY; idx++, iy += blockDim.y)
	{
		val[idx] = gdata[(boffset + iy)*blockDim.x + threadIdx.x];
		data[iy * blockDim.x + threadIdx.x] = val[idx];
	}
	__syncthreads();

	// Start iterative thinning
	int count = 1;
	unsigned int up, down, iteration = 0;
	const int max_iter = 16; // (overlapRows >> 1);

	while ((count > 0) && (iteration < max_iter))
	{
		count = 0;
		iteration++;

		// Eliminate North Boundary Points
#pragma unroll
		for (int iy = threadIdx.y, idx = 0; idx < NBlockY; idx++, iy += blockDim.y)
		{
			ReadUpDownDataMB(data, iy, up, down);
			count += North(up, down, val[idx]);
		}
		WriteNewDataMB<NBlockY>(data, val);

		// Eliminate South Boundary Points
		for (int iy = threadIdx.y, idx = 0; idx < NBlockY; idx++, iy += blockDim.y)
		{
			ReadUpDownDataMB(data, iy, up, down);
			count += South(up, down, val[idx]);
		}
		WriteNewDataMB<NBlockY>(data, val);

		// Eliminate East Boundary Points
		for (int iy = threadIdx.y, idx = 0; idx < NBlockY; idx++, iy += blockDim.y)
		{
			ReadUpDownDataMB(data, iy, up, down);
			count += East(up, down, val[idx]);
		}
		WriteNewDataMB<NBlockY>(data, val);

		// Eliminate West Boundary Points
		for (int iy = threadIdx.y, idx = 0; idx < NBlockY; idx++, iy += blockDim.y)
		{
			ReadUpDownDataMB(data, iy, up, down);
			count += West(up, down, val[idx]);
		}
		WriteNewDataMB<NBlockY>(data, val);

		// Calculate Number of Overall Changes
		count = warpReduce(count);
		if (widx == 0) cmem[wy] = count;

		__syncthreads();
		if (wy == 0)
		{
			if (widx < wl)
				count = cmem[widx];
			else
				count = 0;
			count = warpReduce(count);
			cmem[widx] = count;
		}
		__syncthreads();
		count = cmem[widx];

	} // End Iterative Thinning

	// Write out results
	__syncthreads();
#pragma unroll
	for (int iy = threadIdx.y, idx = 0; idx < NBlockY; idx++, iy += blockDim.y)
	{
		if (((boffset == 0) || (iy >= (overlapRows / 2))) && ((blockIdx.y == (gridDim.y - 1)) || (iy < (NBlockY*blockDim.y - (overlapRows + 1) / 2))))
			odata[(boffset + iy)*blockDim.x + threadIdx.x] = data[iy*blockDim.x + threadIdx.x];
	}

	// Write num Changes
	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		numChanges[blockIdx.y] = count;
	}
}
#endif // !CUDA_THINNING_KERNEL_MULTI_BLOCK_2D_CUH