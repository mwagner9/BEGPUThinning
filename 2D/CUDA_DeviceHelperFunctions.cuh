/**
 * @file CUDA_DeviceHelperFunctions.h
 *
 * @brief This files implements the CUDA kernels for IO tasks, parallel sums and pixel shifting.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#ifndef CUDA_DEVICE_HELPER_FUNCTIONS_CUH
#define CUDA_DEVICE_HELPER_FUNCTIONS_CUH

#define WARP_SIZE 32

__inline__ __device__ unsigned int warpReduce(unsigned int mySum) {

#if CUDART_VERSION >= 9000
	mySum += __shfl_xor_sync(0xffffffff, mySum, 16);
	mySum += __shfl_xor_sync(0xffffffff, mySum, 8);
	mySum += __shfl_xor_sync(0xffffffff, mySum, 4);
	mySum += __shfl_xor_sync(0xffffffff, mySum, 2);
	mySum += __shfl_xor_sync(0xffffffff, mySum, 1);
#else
	mySum += __shfl_xor(mySum, 16);
	mySum += __shfl_xor(mySum, 8);
	mySum += __shfl_xor(mySum, 4);
	mySum += __shfl_xor(mySum, 2);
	mySum += __shfl_xor(mySum, 1);
#endif	
	return mySum;
}

__inline__ __device__ unsigned int ShiftLeft(unsigned int val)
{
#if CUDART_VERSION >= 9000
	return (val << 1 | (__shfl_up_sync(0xffffffff, val, 1, WARP_SIZE) >> 31));
#else
	return (val << 1 | (__shfl_up(val, 1, WARP_SIZE) >> 31));
#endif
}

__inline__ __device__ unsigned int ShiftRight(unsigned int val)
{
#if CUDART_VERSION >= 9000
	return (val >> 1 | (__shfl_down_sync(0xffffffff, val, 1, WARP_SIZE) << 31));
#else
	return (val >> 1 | (__shfl_down(val, 1, WARP_SIZE) << 31));
#endif
}

template <typename T>
__global__ void ExtractBitsKernel(unsigned int *idata, T *odata, unsigned int width, unsigned int height)
{

	// Allocate Shared Memory
	__shared__ volatile unsigned int data[WARP_SIZE][WARP_SIZE];
	__shared__ volatile unsigned int extracted[WARP_SIZE][WARP_SIZE + 1];

	// Calculate Indices
	int lidx = blockDim.x*(blockIdx.x * blockDim.y + threadIdx.y) + threadIdx.x;
	int sidx = blockIdx.x * blockDim.x * blockDim.y * WARP_SIZE;

	// Load Data
	if (lidx < ((width >> 5)*height))
		data[threadIdx.y][threadIdx.x] = idata[lidx];
	__syncthreads();

	// Extract Row by Row
#pragma unroll
	for (int ty = 0; ty < WARP_SIZE; ty++)
	{
		// Extract bits
		extracted[threadIdx.y][threadIdx.x] = (unsigned int)((data[ty][threadIdx.x] >> threadIdx.y) & 0x1);
		__syncthreads();

		// Write to output
		int idx = sidx + (ty*WARP_SIZE + threadIdx.y)*WARP_SIZE + threadIdx.x;
		if (idx < (width*height))
			odata[idx] = extracted[threadIdx.x][threadIdx.y];
		__syncthreads();
	}
}

template<typename T>
__global__ void EncodeBitsKernel(T *iimg, unsigned int *oimg, unsigned int width, unsigned int height)
{
	__shared__ unsigned int data[WARP_SIZE][WARP_SIZE];

	// Calculate indices
	unsigned int tidx = blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int iidx = blockIdx.x * blockDim.x * blockDim.y * WARP_SIZE + tidx;

#pragma unroll
	for (unsigned int k = 0; k < WARP_SIZE; k++, iidx += (blockDim.x*blockDim.y))
	{
		// Read input data
		unsigned int val;
		
		if (iidx < (width*height))
			val = ((unsigned int)iimg[iidx] > 0) << threadIdx.x;
		else
			val = 0;

		// Reduce to single value per Warp
#if CUDART_VERSION >= 9000
		val |= __shfl_xor_sync(0xffffffff, val, 16);
		val |= __shfl_xor_sync(0xffffffff, val, 8);
		val |= __shfl_xor_sync(0xffffffff, val, 4);
		val |= __shfl_xor_sync(0xffffffff, val, 2);
		val |= __shfl_xor_sync(0xffffffff, val, 1);
#else
		val |= __shfl_xor(val, 16);
		val |= __shfl_xor(val, 8);
		val |= __shfl_xor(val, 4);
		val |= __shfl_xor(val, 2);
		val |= __shfl_xor(val, 1);
#endif

		// Write to shared memory
		if (threadIdx.x == 0) data[k][threadIdx.y] = val;
	}

	// Write to global memory
	__syncthreads();
	int idx = (blockIdx.x*blockDim.y*blockDim.x) + tidx;
	if (idx < (width >> 5)*height)
		oimg[idx] = data[threadIdx.y][threadIdx.x];

}


#endif // !CUDA_DEVICE_HELPER_FUNCTIONS_CUH