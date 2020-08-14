/**
 * @file CUDA_ThinningKernel3D_Linear.cuh
 *
 * @brief This files implements the main CUDA kernels to perform 3D thinning.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#ifndef _CUDA_THINNING_KERNEL_3D_LINEAR_CUH_
#define _CUDA_THINNING_KERNEL_3D__LINEAR_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_SubiterationKernels3D.cuh"

#define _USE_MASKED_COUNT_

#define GENERATE_BLOCK_MASK(DIR) \
			GetNeighbors3DZBlock<0>(data, n, val[0]);	\
			count |= DIR##3DLinear(val[0], n, pv[0]); \
			GetNeighbors3DZBlock<1>(data, n, val[1]);	\
			count |= DIR##3DLinear(val[1], n, pv[1]); \
			GetNeighbors3DZBlock<2>(data, n, val[2]);	\
			count |= DIR##3DLinear(val[2], n, pv[2]); \
			GetNeighbors3DZBlock<3>(data, n, val[3]);	\
			count |= DIR##3DLinear(val[3], n, pv[3]); \
			GetNeighbors3DZBlock<4>(data, n, val[4]);	\
			count |= DIR##3DLinear(val[4], n, pv[4]); \
			GetNeighbors3DZBlock<5>(data, n, val[5]);	\
			count |= DIR##3DLinear(val[5], n, pv[5]); \
			GetNeighbors3DZBlock<6>(data, n, val[6]);	\
			count |= DIR##3DLinear(val[6], n, pv[6]); \
			GetNeighbors3DZBlock<7>(data, n, val[7]);	\
			count |= DIR##3DLinear(val[7], n, pv[7]); 

__inline__ __device__ unsigned int warpReduce3DLinear(unsigned int mySum) 
{

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

template<int bi>
__inline__ __device__ void GetNeighbors3DZBlock(volatile unsigned int data[][2][64], unsigned int *n, unsigned int val)
{
	int xoff = (bi % 2) * 32;
	int yoff = ((bi >> 1) % 2);
	int zoff = ((bi >> 2) % 2) * 32;

	// Top Layer
	if (threadIdx.z == 0 && zoff == 0)
	{
#pragma unroll
		for (int i = 0; i < 8; i++) n[i] = 0;
	}
	else
	{

		n[4] = data[threadIdx.z - 1 + zoff][yoff][threadIdx.x + xoff]; // Up

		if (threadIdx.x == 0 && xoff == 0)
		{
			n[0] = 0; n[3] = 0; n[6] = 0;
		}
		else
		{
			n[3] = data[threadIdx.z - 1 + zoff][yoff][threadIdx.x + xoff - 1];

			n[0] = (n[3] << 1) | ((yoff > 0) ? (data[threadIdx.z - 1 + zoff][yoff - 1][threadIdx.x + xoff - 1] >> 31) : 0);
			n[6] = (n[3] >> 1) | ((yoff == 0) ? (data[threadIdx.z - 1 + zoff][yoff + 1][threadIdx.x + xoff - 1] << 31) : 0);
		}

		if (threadIdx.x == 31 && xoff > 0)
		{
			n[2] = 0; n[5] = 0; n[8] = 0;
		}
		else
		{
			n[5] = data[threadIdx.z - 1 + zoff][yoff][threadIdx.x + xoff + 1];

			n[2] = (n[5] << 1) | ((yoff > 0) ? (data[threadIdx.z - 1 + zoff][yoff - 1][threadIdx.x + xoff + 1] >> 31) : 0);
			n[8] = (n[5] >> 1) | ((yoff == 0) ? (data[threadIdx.z - 1 + zoff][yoff + 1][threadIdx.x + xoff + 1] << 31) : 0);
		}

		n[1] = (n[4] << 1) | ((yoff > 0) ? (data[threadIdx.z - 1 + zoff][yoff - 1][threadIdx.x + xoff] >> 31) : 0);
		n[7] = (n[4] >> 1) | ((yoff == 0) ? (data[threadIdx.z - 1 + zoff][yoff + 1][threadIdx.x + xoff] << 31) : 0);

	}

	// Mid Layer
	if (threadIdx.x == 0 && xoff == 0)
	{
		n[9] = 0; n[12] = 0; n[14] = 0;
	}
	else
	{
		n[12] = data[threadIdx.z + zoff][yoff][threadIdx.x + xoff - 1];

		n[9] = (n[12] << 1) | ((yoff > 0) ? (data[threadIdx.z + zoff][yoff - 1][threadIdx.x + xoff - 1] >> 31) : 0);
		n[14] = (n[12] >> 1) | ((yoff == 0) ? (data[threadIdx.z + zoff][yoff + 1][threadIdx.x + xoff - 1] << 31) : 0);
	}

	if (threadIdx.x == 31 && xoff > 0)
	{
		n[11] = 0; n[13] = 0; n[16] = 0;
	}
	else
	{
		n[13] = data[threadIdx.z + zoff][yoff][threadIdx.x + xoff + 1];

		n[11] = (n[13] << 1) | ((yoff > 0) ? (data[threadIdx.z + zoff][yoff - 1][threadIdx.x + xoff + 1] >> 31) : 0);
		n[16] = (n[13] >> 1) | ((yoff == 0) ? (data[threadIdx.z + zoff][yoff + 1][threadIdx.x + xoff + 1] << 31) : 0);
	}

	n[10] = (val << 1) | ((yoff > 0) ? (data[threadIdx.z + zoff][yoff - 1][threadIdx.x + xoff] >> 31) : 0);
	n[15] = (val >> 1) | ((yoff == 0) ? (data[threadIdx.z + zoff][yoff + 1][threadIdx.x + xoff] << 31) : 0);

	// Bottom Layer
	if (threadIdx.z == 31 && zoff > 0)
	{
#pragma unroll
		for (int i = 17; i < 26; i++) n[i] = 0;
	}
	else
	{

		n[21] = data[threadIdx.z + 1 + zoff][yoff][threadIdx.x + xoff]; // Down

		if (threadIdx.x == 0 && xoff == 0)
		{
			n[17] = 0; n[20] = 0; n[23] = 0;
		}
		else
		{
			n[20] = data[threadIdx.z + 1 + zoff][yoff][threadIdx.x + xoff - 1];

			n[17] = (n[20] << 1) | ((yoff > 0) ? (data[threadIdx.z + 1 + zoff][yoff - 1][threadIdx.x + xoff - 1] >> 31) : 0);
			n[23] = (n[20] >> 1) | ((yoff == 0) ? (data[threadIdx.z + 1 + zoff][yoff + 1][threadIdx.x + xoff - 1] << 31) : 0);
		}

		if (threadIdx.x == 31 && xoff > 0)
		{
			n[19] = 0; n[22] = 0; n[25] = 0;
		}
		else
		{
			n[22] = data[threadIdx.z + 1 + zoff][yoff][threadIdx.x + xoff + 1];

			n[19] = (n[22] << 1) | ((yoff > 0) ? (data[threadIdx.z + 1 + zoff][yoff - 1][threadIdx.x + xoff + 1] >> 31) : 0);
			n[25] = (n[22] >> 1) | ((yoff == 0) ? (data[threadIdx.z + 1 + zoff][yoff + 1][threadIdx.x + xoff + 1] << 31) : 0);
		}

		n[18] = (n[21] << 1) | ((yoff > 0) ? (data[threadIdx.z + 1 + zoff][yoff - 1][threadIdx.x + xoff] >> 31) : 0);
		n[24] = (n[21] >> 1) | ((yoff == 0) ? (data[threadIdx.z + 1 + zoff][yoff + 1][threadIdx.x + xoff] << 31) : 0);
	}




}

__inline__ __device__ void GetNeighbors3DZ(volatile unsigned int data[][32], unsigned int *n, unsigned int val)
{
	// Z-Dimension
	n[4] = (threadIdx.z == 0) ? 0 : data[threadIdx.z - 1][threadIdx.x]; // Up
	n[21] = (threadIdx.z == 31) ? 0 : data[threadIdx.z + 1][threadIdx.x]; // Down

	// X-Dimension
#if CUDART_VERSION >= 9000	
	n[3] = __shfl_up_sync(0xffffffff, n[4], 1, 32);
	n[5] = __shfl_down_sync(0xffffffff, n[4], 1, 32);
	n[12] = __shfl_up_sync(0xffffffff, val, 1, 32);
	n[13] = __shfl_down_sync(0xffffffff, val, 1, 32);
	n[20] = __shfl_up_sync(0xffffffff, n[21], 1, 32);
	n[22] = __shfl_down_sync(0xffffffff, n[21], 1, 32);
#else
	n[3] = __shfl_up(n[4], 1, 32);
	n[5] = __shfl_down(n[4], 1, 32);
	n[12] = __shfl_up(val, 1, 32);
	n[13] = __shfl_down(val, 1, 32);
	n[20] = __shfl_up(n[21], 1, 32);
	n[22] = __shfl_down(n[21], 1, 32);
#endif
	
	// Y-Dimension
	n[0] = (n[3] >> 1);
	n[6] = (n[3] << 1);
	n[1] = (n[4] >> 1);
	n[7] = (n[4] << 1);
	n[2] = (n[5] >> 1);
	n[8] = (n[5] << 1);

	n[9] = (n[12] >> 1);
	n[14] = (n[12] << 1);
	n[10] = (val >> 1);
	n[15] = (val << 1);
	n[11] = (n[13] >> 1);
	n[16] = (n[13] << 1);

	n[17] = (n[20] >> 1);
	n[23] = (n[20] << 1);
	n[18] = (n[21] >> 1);
	n[24] = (n[21] << 1);
	n[19] = (n[22] >> 1);
	n[25] = (n[22] << 1);
}


__inline__ __device__ int Up3DLinear(unsigned int &val, unsigned int *n, unsigned int prev = 0)
{
	// Interim Results
	unsigned int tmp1 = n[9] | n[10] | n[11] | n[12] | n[13] | n[14] | n[15] | n[16] | n[17] | n[19] | n[23] | n[25];	//
	unsigned int tmp2 = n[20] | n[22];																					//
	unsigned int tmp3 = n[18] | n[24];																					//
	unsigned int t1 = ~n[15] & ~n[16] & n[18] & ~n[24] & ~n[25];														//
	unsigned int t2 = ~n[11] & ~n[10] & ~n[18] & ~n[19] & n[24];														//
	unsigned int t3 = ~n[13] & ~n[22] & n[20];																			//
	unsigned int t4 = ~n[12] & ~n[20] & n[22];																			//
	unsigned int t6 = ~n[14] & ~n[23];																					//
	unsigned int t5 = t4 & t6;																							//
	t4 &= ~n[9] & ~n[17];																								//

	unsigned int cmask1 = (tmp1 | tmp2 | tmp3); // Finish M1															//
	unsigned int cmask5 = ((tmp1 | tmp2) & ((~n[14] & ~n[23] & t1) | (~n[9] & ~n[17] & t2))) |
		((tmp1 | tmp3) & ((t4 & t6) | (~n[11] & ~n[16] & ~n[19] & ~n[25] & t3)));										// 
	unsigned int cmask6 = (~n[24] & ~n[15] & n[18] & t5) |
		(~n[18] & ~n[10] & n[24] & t4) |
		(t3 & t2) |
		(t3 & t1);																										// 
	unsigned int cmask4 = (~n[0] & ~n[6] & ((n[2] & n[11] & ~n[8]) | (n[8] & n[16] & ~n[2]))) | (~n[2] & ~n[8] & ((n[0] & n[9] & ~n[6]) | (n[6] & n[14] & ~n[0])));						// 
	unsigned int cmask2 = (~n[1] & ~n[2] & ~n[5] & ((~n[0] & ~n[3] & n[15]) | (~n[7] & ~n[8] & n[12]))) | (~n[7] & ~n[6] & ~n[3] & ((~n[5] & ~n[8] & n[10]) | (~n[0] & ~n[1] & n[13]))); // 
	unsigned int cmask3 = (~n[3] & n[13] & ((~n[7] & ~n[6] & n[10]) | (~n[0] & ~n[1] & n[15]))) | (~n[5] & n[12] & ((~n[1] & ~n[2] & n[15]) | (~n[7] & ~n[8] & n[10])));					//

	// Check Combined
	unsigned int combined = ~n[4] & // North Neighbor is always zero
		(((~n[1] & ~n[3] & ~n[5] & ~n[7]) & // North Cross for M1, M4, M5 and M6
		((n[21] & cmask4) | // Finish M4
		((~n[0] & ~n[2] & ~n[6] & ~n[8]) & // North Diagonals for M1, M5 and M6
		((n[21] & cmask1) | (~n[21] & (cmask5 | cmask6)))))) | // Finish M1, M5, and M6
		(n[21] & (cmask2 | cmask3))); // Add masks M2 and M3			

	combined &= (val & ~prev);
	val &= ~combined;

	return combined;

} // End Up3D

__inline__ __device__ int Down3DLinear(unsigned int &val, unsigned int *n, unsigned int prev = 0)
{
	// Interim Results
	unsigned int tmp1 = n[11] | n[10] | n[9] | n[13] | n[12] | n[16] | n[15] | n[14] | n[2] | n[0] | n[8] | n[6];	//
	unsigned int tmp2 = n[5] | n[3];																					//
	unsigned int tmp3 = n[1] | n[7];																					//
	unsigned int t1 = ~n[15] & ~n[14] & n[1] & ~n[7] & ~n[6];														//
	unsigned int t2 = ~n[9] & ~n[10] & ~n[1] & ~n[0] & n[7];														//
	unsigned int t3 = ~n[12] & ~n[3] & n[5];																			//
	unsigned int t4 = ~n[13] & ~n[5] & n[3];																			//
	unsigned int t6 = ~n[16] & ~n[8];																					//
	unsigned int t5 = t4 & t6;																							//
	t4 &= ~n[11] & ~n[2];																								//

	unsigned int cmask1 = (tmp1 | tmp2 | tmp3); // Finish M1															//
	unsigned int cmask5 = ((tmp1 | tmp2) & ((~n[16] & ~n[8] & t1) | (~n[11] & ~n[2] & t2))) |
		((tmp1 | tmp3) & ((t4 & t6) | (~n[9] & ~n[14] & ~n[0] & ~n[6] & t3)));										// 
	unsigned int cmask6 = (~n[7] & ~n[15] & n[1] & t5) |
		(~n[1] & ~n[10] & n[7] & t4) |
		(t3 & t2) |
		(t3 & t1);																										// 
	unsigned int cmask4 = (~n[19] & ~n[25] & ((n[17] & n[9] & ~n[23]) | (n[23] & n[14] & ~n[17]))) | (~n[17] & ~n[23] & ((n[19] & n[11] & ~n[25]) | (n[25] & n[16] & ~n[19])));						// 
	unsigned int cmask2 = (~n[18] & ~n[17] & ~n[20] & ((~n[19] & ~n[22] & n[15]) | (~n[24] & ~n[23] & n[13]))) | (~n[24] & ~n[25] & ~n[22] & ((~n[20] & ~n[23] & n[10]) | (~n[19] & ~n[18] & n[12]))); // 
	unsigned int cmask3 = (~n[22] & n[12] & ((~n[24] & ~n[25] & n[10]) | (~n[19] & ~n[18] & n[15]))) | (~n[20] & n[13] & ((~n[18] & ~n[17] & n[15]) | (~n[24] & ~n[23] & n[10])));					//

	// Check Combined
	unsigned int combined = ~n[21] & // North Neighbor is always zero
		(((~n[18] & ~n[22] & ~n[20] & ~n[24]) & // North Cross for M1, M4, M5 and M6
		((n[4] & cmask4) | // Finish M4
		((~n[19] & ~n[17] & ~n[25] & ~n[23]) & // North Diagonals for M1, M5 and M6
		((n[4] & cmask1) | (~n[4] & (cmask5 | cmask6)))))) | // Finish M1, M5, and M6
		(n[4] & (cmask2 | cmask3))); // Add masks M2 and M3			

	combined &= (val & ~prev);
	val &= ~combined;

	return combined;

} // End Down3D

__inline__ __device__ int South3DLinear(unsigned int &val, unsigned int *n, unsigned int prev = 0)
{
	// Interim Results
	unsigned int tmp1 = n[3] | n[4] | n[5] | n[12] | n[13] | n[20] | n[21] | n[22] | n[0] | n[2] | n[17] | n[19];	//
	unsigned int tmp2 = n[9] | n[11];																					//
	unsigned int tmp3 = n[1] | n[18];																					//
	unsigned int t1 = ~n[21] & ~n[22] & n[1] & ~n[18] & ~n[19];														//
	unsigned int t2 = ~n[5] & ~n[4] & ~n[1] & ~n[2] & n[18];														//
	unsigned int t3 = ~n[13] & ~n[11] & n[9];																			//
	unsigned int t4 = ~n[12] & ~n[9] & n[11];																			//
	unsigned int t6 = ~n[20] & ~n[17];																					//
	unsigned int t5 = t4 & t6;																							//
	t4 &= ~n[3] & ~n[0];																								//

	unsigned int cmask1 = (tmp1 | tmp2 | tmp3); // Finish M1															//
	unsigned int cmask5 = ((tmp1 | tmp2) & ((~n[20] & ~n[17] & t1) | (~n[3] & ~n[0] & t2))) |
		((tmp1 | tmp3) & ((t4 & t6) | (~n[5] & ~n[22] & ~n[2] & ~n[19] & t3)));										// 
	unsigned int cmask6 = (~n[18] & ~n[21] & n[1] & t5) |
		(~n[1] & ~n[4] & n[18] & t4) |
		(t3 & t2) |
		(t3 & t1);																										// 
	unsigned int cmask4 = (~n[6] & ~n[23] & ((n[8] & n[5] & ~n[25]) | (n[25] & n[22] & ~n[8]))) | (~n[8] & ~n[25] & ((n[6] & n[3] & ~n[23]) | (n[23] & n[20] & ~n[6])));						// 
	unsigned int cmask2 = (~n[7] & ~n[8] & ~n[16] & ((~n[6] & ~n[14] & n[21]) | (~n[24] & ~n[25] & n[12]))) | (~n[24] & ~n[23] & ~n[14] & ((~n[16] & ~n[25] & n[4]) | (~n[6] & ~n[7] & n[13]))); // 
	unsigned int cmask3 = (~n[14] & n[13] & ((~n[24] & ~n[23] & n[4]) | (~n[6] & ~n[7] & n[21]))) | (~n[16] & n[12] & ((~n[7] & ~n[8] & n[21]) | (~n[24] & ~n[25] & n[4])));					//

	// Check Combined
	unsigned int combined = ~n[15] & // North Neighbor is always zero
		(((~n[7] & ~n[14] & ~n[16] & ~n[24]) & // North Cross for M1, M4, M5 and M6
		((n[10] & cmask4) | // Finish M4
		((~n[6] & ~n[8] & ~n[23] & ~n[25]) & // North Diagonals for M1, M5 and M6
		((n[10] & cmask1) | (~n[10] & (cmask5 | cmask6)))))) | // Finish M1, M5, and M6
		(n[10] & (cmask2 | cmask3))); // Add masks M2 and M3			

	combined &= (val & ~prev);
	val &= ~combined;

	return combined;

} // End North3D

__inline__ __device__ int North3DLinear(unsigned int &val, unsigned int *n, unsigned int prev = 0)
{
	// Interim Results
	unsigned int tmp1 = n[20] | n[21] | n[22] | n[12] | n[13] | n[3] | n[4] | n[5] | n[23] | n[25] | n[6] | n[8];	//
	unsigned int tmp2 = n[14] | n[16];																					//
	unsigned int tmp3 = n[24] | n[7];																					//
	unsigned int t1 = ~n[4] & ~n[5] & n[24] & ~n[7] & ~n[8];														//
	unsigned int t2 = ~n[22] & ~n[21] & ~n[24] & ~n[25] & n[7];														//
	unsigned int t3 = ~n[13] & ~n[16] & n[14];																			//
	unsigned int t4 = ~n[12] & ~n[14] & n[16];																			//
	unsigned int t6 = ~n[3] & ~n[6];																					//
	unsigned int t5 = t4 & t6;																							//
	t4 &= ~n[20] & ~n[23];																								//

	unsigned int cmask1 = (tmp1 | tmp2 | tmp3); // Finish M1															//
	unsigned int cmask5 = ((tmp1 | tmp2) & ((~n[3] & ~n[6] & t1) | (~n[20] & ~n[23] & t2))) |
		((tmp1 | tmp3) & ((t4 & t6) | (~n[22] & ~n[5] & ~n[25] & ~n[8] & t3)));										// 
	unsigned int cmask6 = (~n[7] & ~n[4] & n[24] & t5) |
		(~n[24] & ~n[21] & n[7] & t4) |
		(t3 & t2) |
		(t3 & t1);																										// 
	unsigned int cmask4 = (~n[17] & ~n[0] & ((n[19] & n[22] & ~n[2]) | (n[2] & n[5] & ~n[19]))) | (~n[19] & ~n[2] & ((n[17] & n[20] & ~n[0]) | (n[0] & n[3] & ~n[17])));						// 
	unsigned int cmask2 = (~n[18] & ~n[19] & ~n[11] & ((~n[17] & ~n[9] & n[4]) | (~n[1] & ~n[2] & n[12]))) | (~n[1] & ~n[0] & ~n[9] & ((~n[11] & ~n[2] & n[21]) | (~n[17] & ~n[18] & n[13]))); // 
	unsigned int cmask3 = (~n[9] & n[13] & ((~n[1] & ~n[0] & n[21]) | (~n[17] & ~n[18] & n[4]))) | (~n[11] & n[12] & ((~n[18] & ~n[19] & n[4]) | (~n[1] & ~n[2] & n[21])));					//

	// Check Combined
	unsigned int combined = ~n[10] & // North Neighbor is always zero
		(((~n[18] & ~n[9] & ~n[11] & ~n[1]) & // North Cross for M1, M4, M5 and M6
		((n[15] & cmask4) | // Finish M4
		((~n[17] & ~n[19] & ~n[0] & ~n[2]) & // North Diagonals for M1, M5 and M6
		((n[15] & cmask1) | (~n[15] & (cmask5 | cmask6)))))) | // Finish M1, M5, and M6
		(n[15] & (cmask2 | cmask3))); // Add masks M2 and M3			

	combined &= (val & ~prev);
	val &= ~combined;

	return combined;

} // End South3D

__inline__ __device__ int East3DLinear(unsigned int &val, unsigned int *n, unsigned int prev = 0)
{
	// Interim Results
	unsigned int tmp1 = n[18] | n[10] | n[1] | n[21] | n[4] | n[24] | n[15] | n[7] | n[19] | n[2] | n[25] | n[8];	//
	unsigned int tmp2 = n[22] | n[5];																					//
	unsigned int tmp3 = n[11] | n[16];																					//
	unsigned int t1 = ~n[15] & ~n[7] & n[11] & ~n[16] & ~n[8];														//
	unsigned int t2 = ~n[1] & ~n[10] & ~n[11] & ~n[2] & n[16];														//
	unsigned int t3 = ~n[4] & ~n[5] & n[22];																			//
	unsigned int t4 = ~n[21] & ~n[22] & n[5];																			//
	unsigned int t6 = ~n[24] & ~n[25];																					//
	unsigned int t5 = t4 & t6;																							//
	t4 &= ~n[18] & ~n[19];																								//

	unsigned int cmask1 = (tmp1 | tmp2 | tmp3); // Finish M1															//
	unsigned int cmask5 = ((tmp1 | tmp2) & ((~n[24] & ~n[25] & t1) | (~n[18] & ~n[19] & t2))) |
		((tmp1 | tmp3) & ((t4 & t6) | (~n[1] & ~n[7] & ~n[2] & ~n[8] & t3)));										// 
	unsigned int cmask6 = (~n[16] & ~n[15] & n[11] & t5) |
		(~n[11] & ~n[10] & n[16] & t4) |
		(t3 & t2) |
		(t3 & t1);																										// 
	unsigned int cmask4 = (~n[17] & ~n[23] & ((n[0] & n[1] & ~n[6]) | (n[6] & n[7] & ~n[0]))) | (~n[0] & ~n[6] & ((n[17] & n[18] & ~n[23]) | (n[23] & n[24] & ~n[17])));						// 
	unsigned int cmask2 = (~n[9] & ~n[0] & ~n[3] & ((~n[17] & ~n[20] & n[15]) | (~n[14] & ~n[6] & n[21]))) | (~n[14] & ~n[23] & ~n[20] & ((~n[3] & ~n[6] & n[10]) | (~n[17] & ~n[9] & n[4]))); // 
	unsigned int cmask3 = (~n[20] & n[4] & ((~n[14] & ~n[23] & n[10]) | (~n[17] & ~n[9] & n[15]))) | (~n[3] & n[21] & ((~n[9] & ~n[0] & n[15]) | (~n[14] & ~n[6] & n[10])));					//

	// Check Combined
	unsigned int combined = ~n[12] & // North Neighbor is always zero
		(((~n[9] & ~n[20] & ~n[3] & ~n[14]) & // North Cross for M1, M4, M5 and M6
		((n[13] & cmask4) | // Finish M4
		((~n[17] & ~n[0] & ~n[23] & ~n[6]) & // North Diagonals for M1, M5 and M6
		((n[13] & cmask1) | (~n[13] & (cmask5 | cmask6)))))) | // Finish M1, M5, and M6
		(n[13] & (cmask2 | cmask3))); // Add masks M2 and M3			

	combined &= (val & ~prev);
	val &= ~combined;

	return combined;

} // End East3D

__inline__ __device__ int West3DLinear(unsigned int &val, unsigned int *n, unsigned int prev = 0)
{
	// Interim Results
	unsigned int tmp1 = n[1] | n[10] | n[18] | n[4] | n[21] | n[7] | n[15] | n[24] | n[0] | n[17] | n[6] | n[23];	//
	unsigned int tmp2 = n[3] | n[20];																					//
	unsigned int tmp3 = n[9] | n[14];																					//
	unsigned int t1 = ~n[15] & ~n[24] & n[9] & ~n[14] & ~n[23];														//
	unsigned int t2 = ~n[18] & ~n[10] & ~n[9] & ~n[17] & n[14];														//
	unsigned int t3 = ~n[21] & ~n[20] & n[3];																			//
	unsigned int t4 = ~n[4] & ~n[3] & n[20];																			//
	unsigned int t6 = ~n[7] & ~n[6];																					//
	unsigned int t5 = t4 & t6;																							//
	t4 &= ~n[1] & ~n[0];																								//

	unsigned int cmask1 = (tmp1 | tmp2 | tmp3); // Finish M1															//
	unsigned int cmask5 = ((tmp1 | tmp2) & ((~n[7] & ~n[6] & t1) | (~n[1] & ~n[0] & t2))) |
		((tmp1 | tmp3) & ((t4 & t6) | (~n[18] & ~n[24] & ~n[17] & ~n[23] & t3)));										// 
	unsigned int cmask6 = (~n[14] & ~n[15] & n[9] & t5) |
		(~n[9] & ~n[10] & n[14] & t4) |
		(t3 & t2) |
		(t3 & t1);																										// 
	unsigned int cmask4 = (~n[2] & ~n[8] & ((n[19] & n[18] & ~n[25]) | (n[25] & n[24] & ~n[19]))) | (~n[19] & ~n[25] & ((n[2] & n[1] & ~n[8]) | (n[8] & n[7] & ~n[2])));						// 
	unsigned int cmask2 = (~n[11] & ~n[19] & ~n[22] & ((~n[2] & ~n[5] & n[15]) | (~n[16] & ~n[25] & n[4]))) | (~n[16] & ~n[8] & ~n[5] & ((~n[22] & ~n[25] & n[10]) | (~n[2] & ~n[11] & n[21]))); // 
	unsigned int cmask3 = (~n[5] & n[21] & ((~n[16] & ~n[8] & n[10]) | (~n[2] & ~n[11] & n[15]))) | (~n[22] & n[4] & ((~n[11] & ~n[19] & n[15]) | (~n[16] & ~n[25] & n[10])));					//

	// Check Combined
	unsigned int combined = ~n[13] & // North Neighbor is always zero
		(((~n[11] & ~n[5] & ~n[22] & ~n[16]) & // North Cross for M1, M4, M5 and M6
		((n[12] & cmask4) | // Finish M4
		((~n[2] & ~n[19] & ~n[8] & ~n[25]) & // North Diagonals for M1, M5 and M6
		((n[12] & cmask1) | (~n[12] & (cmask5 | cmask6)))))) | // Finish M1, M5, and M6
		(n[12] & (cmask2 | cmask3))); // Add masks M2 and M3			

	combined &= (val & ~prev);
	val &= ~combined;

	return combined;

} // End West3D

/*
__device__ void PrintBlock(volatile unsigned int data[][32], unsigned int slice)
{
	__syncthreads();
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		//for (int iz = 13; iz < 21; iz++) //blockDim.y
		{
			printf("Slice %i:\n", slice);
			for (int iy = 11; iy < 21; iy++)
			{
				unsigned int val = data[slice][iy];
				for (int ix = 11; ix < 21; ix++)
				{
					printf("%i", (val >> ix) & 1);
				}
				printf("\n");
			}
			printf("\n");
		}
	}
	__syncthreads();
}*/

__device__ void PrintPixelBlock(unsigned int n[], unsigned int val[], int x, int y, int z, volatile unsigned int data[][2][64])
{
	int bx = (x < 16) ? 0 : ((x - 16) >> 5);
	int tx = (x - 32 * bx);
	int ix = 0;
	if (tx >= 32) {
		ix = 1; tx -= 32;
	}
	int bz = (z < 16) ? 0 : ((z - 16) >> 5);
	int tz = (z - 32 * bz);
	int iz = 0;
	if (tz >= 32) {
		iz = 1; tz -= 32;
	}

	int by = (y < 16) ? 0 : ((y - 16) >> 5);
	int ty = (y - 32 * by);
	int iy = 0;
	if (ty >= 32) {
		iy = 1; ty -= 32;
	}

	__syncthreads();

	if ((blockIdx.x == bx) && (threadIdx.x == tx) && (blockIdx.z == bz) && (threadIdx.z == tz) && (blockIdx.y == by))
	{
		int bit = ty;
		int idx = 4 * iz + 2 * iy + ix;
		if (idx == 0) GetNeighbors3DZBlock<0>(data, n, val[idx]);
		else if (idx == 1) GetNeighbors3DZBlock<1>(data, n, val[idx]);
		else if (idx == 2) GetNeighbors3DZBlock<2>(data, n, val[idx]);
		else if (idx == 3) GetNeighbors3DZBlock<3>(data, n, val[idx]);
		else if (idx == 4) GetNeighbors3DZBlock<4>(data, n, val[idx]);
		else if (idx == 5) GetNeighbors3DZBlock<5>(data, n, val[idx]);
		else if (idx == 6) GetNeighbors3DZBlock<6>(data, n, val[idx]);
		else if (idx == 7) GetNeighbors3DZBlock<7>(data, n, val[idx]);

		printf("  %i  %i  %i\n", (n[0] >> bit) & 1, (n[1] >> bit) & 1, (n[2] >> bit) & 1);
		printf(" %i  %i  %i\n", (n[3] >> bit) & 1, (n[4] >> bit) & 1, (n[5] >> bit) & 1);
		printf("%i  %i  %i\n\n", (n[6] >> bit) & 1, (n[7] >> bit) & 1, (n[8] >> bit) & 1);

		printf("  %i  %i  %i\n", (n[9] >> bit) & 1, (n[10] >> bit) & 1, (n[11] >> bit) & 1);
		printf(" %i  %i  %i\n", (n[12] >> bit) & 1, (val[idx] >> bit) & 1, (n[13] >> bit) & 1);
		printf("%i  %i  %i\n\n", (n[14] >> bit) & 1, (n[15] >> bit) & 1, (n[16] >> bit) & 1);

		printf("  %i  %i  %i\n", (n[17] >> bit) & 1, (n[18] >> bit) & 1, (n[19] >> bit) & 1);
		printf(" %i  %i  %i\n", (n[20] >> bit) & 1, (n[21] >> bit) & 1, (n[22] >> bit) & 1);
		printf("%i  %i  %i\n\n", (n[23] >> bit) & 1, (n[24] >> bit) & 1, (n[25] >> bit) & 1);

		int tmp = data[1][1][43]; 
		tmp = (tmp >> bit) & 1;
		printf("block(%i, %i, %i); thread(%i, %i, %i); val = %i; idx = %i; val2 = %i\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, ty + iy * 32, threadIdx.z, (data[threadIdx.z + iz*blockDim.z][iy][threadIdx.x + ix*blockDim.x] >> bit) & 1, idx, tmp);
	}
	__syncthreads();
}

__global__ void __launch_bounds__(1024) ThinningKernel3DLinear(unsigned char *gdata, unsigned char *odata, int *numChanges)
{
	__shared__ volatile unsigned int data[32][32];
	__shared__ volatile int cmem[32];

	// Load Data
	unsigned int height = ((gridDim.y + 1) << 1);
	unsigned int width = ((gridDim.x + 1) << 4);
	int iz = ((blockIdx.z << 4) + threadIdx.z);
	int iy = (blockIdx.y << 1);
	int ix = ((blockIdx.x << 4) + threadIdx.x);
	int lidx = (iz * height + iy) * width + ix;
	unsigned int val = gdata[lidx] | (gdata[lidx + width] << 8) | (gdata[lidx + 2 * width] << 16) | (gdata[lidx + 3 * width] << 24);
	unsigned int oval = val;

	data[threadIdx.z][threadIdx.x] = val;
	__syncthreads();

	// Start iterative thinning
	unsigned int count = 1, iteration = 0;
	unsigned int n[26];

	while ((count > 0) && (iteration < 8))
	{
		iteration++;

		// Calculate First Round Without Deleting
		//unsigned int pd = val, pn = val, ps = val, pe = val, pw = val;
		GetNeighbors3DZ(data, n, val);
		unsigned int pd = Down3D(val, n);
		unsigned int pn = North3D(val, n);
		unsigned int ps = South3D(val, n);
		unsigned int pe = East3D(val, n);
		unsigned int pw = West3D(val, n);
		unsigned int ival = val;

		// Eliminate Up Boundary Points
		val = Up3D(val, n);
		data[threadIdx.z][threadIdx.x] = val;
		__syncthreads();

		// Eliminate North Boundary Points
		GetNeighbors3DZ(data, n, val);
		val = North3D(val, n, pn);
		data[threadIdx.z][threadIdx.x] = val;
		__syncthreads();

		// Eliminate East Boundary Points
		GetNeighbors3DZ(data, n, val);
		val = East3D(val, n, pe);
		data[threadIdx.z][threadIdx.x] = val;
		__syncthreads();

		// Eliminate Down Boundary Points
		GetNeighbors3DZ(data, n, val);
		val = Down3D(val, n, pd);
		data[threadIdx.z][threadIdx.x] = val;
		__syncthreads();

		// Eliminate South Boundary Points
		GetNeighbors3DZ(data, n, val);
		val = South3D(val, n, ps);
		data[threadIdx.z][threadIdx.x] = val;
		__syncthreads();

		// Eliminate West Boundary Points
		GetNeighbors3DZ(data, n, val);
		val = West3D(val, n, pw);
		data[threadIdx.z][threadIdx.x] = val;

#ifdef _USE_MASKED_COUNT_		
		if (iteration == 8)
		{
			count = (val ^ oval);
			if (((blockIdx.x > 0) && (threadIdx.x < 8)) || ((blockIdx.z > 0) && (threadIdx.z < 8)) || ((blockIdx.x < (gridDim.x - 1)) && (threadIdx.x >= 24)) || ((blockIdx.z < (gridDim.z - 1)) && (threadIdx.z >= 24)))
			{
				count = 0;
			}
			else
			{
				if (blockIdx.y > 0) count &= 0xffff0000;
				if (blockIdx.y < (gridDim.y - 1)) count &= 0x0000ffff;
			}
			count = (count == 0) ? 0 : 1;
		}
		else
		{
			count = (val != ival) ? 1 : 0;
		}
#else
		count = (val != ival) ? 1 : 0;
#endif	

		// Calculate Number of Overall Changes
		count = warpReduce3DLinear(count);
		if (threadIdx.x == 0) cmem[threadIdx.z] = count;

		__syncthreads();
		if (threadIdx.z == 0)
		{
			count = cmem[threadIdx.x];
			count = warpReduce3DLinear(count);
			cmem[threadIdx.x] = count;
		}
		__syncthreads();
		count = cmem[threadIdx.x];

	} // End Iterative Thinning

	// Write out results
	__syncthreads();
	if (((blockIdx.x == 0) || (threadIdx.x >= 8)) && ((blockIdx.z == 0) || (threadIdx.z >= 8)) && ((blockIdx.x == (gridDim.x - 1)) || (threadIdx.x < 24)) && ((blockIdx.z == (gridDim.z - 1)) || (threadIdx.z < 24)))
	{
		odata[lidx + width] = (unsigned char)(val >> 8);
		odata[lidx + 2 * width] = (unsigned char)(val >> 16);

		if (blockIdx.y == 0)
		{
			odata[lidx] = (unsigned char)(val);
		}
		else if (blockIdx.y == (gridDim.y - 1))
		{
			odata[lidx + 3 * width] = (unsigned char)(val >> 24);
		}
	}

	// Write num Changes
	if ((threadIdx.x == 0) && (threadIdx.z == 0))
	{
		numChanges[(gridDim.y*blockIdx.z + blockIdx.y)*gridDim.x + blockIdx.x] = count;
	}


}

__inline__ __device__ void WriteShareData(volatile unsigned int data[][2][64], unsigned int val[8])
{
	data[threadIdx.z][0][threadIdx.x] = val[0];
	data[threadIdx.z][0][threadIdx.x + blockDim.x] = val[1];
	data[threadIdx.z][1][threadIdx.x] = val[2];
	data[threadIdx.z][1][threadIdx.x + blockDim.x] = val[3];
	data[blockDim.z + threadIdx.z][0][threadIdx.x] = val[4];
	data[blockDim.z + threadIdx.z][0][threadIdx.x + blockDim.x] = val[5];
	data[blockDim.z + threadIdx.z][1][threadIdx.x] = val[6];
	data[blockDim.z + threadIdx.z][1][threadIdx.x + blockDim.x] = val[7];
}

template< int BI >
__inline__ __device__ void PreTest(unsigned int pv[], unsigned int val[], volatile unsigned int data[][2][64], unsigned int n[26])
{
	pv[BI] = val[BI];
	GetNeighbors3DZBlock<BI>(data, n, val[BI]);

	pv[BI] = Up3D(val[BI], n);
	pv[BI] &= North3D(val[BI], n);
	pv[BI] &= East3D(val[BI], n);
	pv[BI] &= Down3D(val[BI], n);
	pv[BI] &= South3D(val[BI], n);
	pv[BI] &= West3D(val[BI], n);
}

__global__ void __launch_bounds__(1024) ThinningKernel3DLinearBlock(unsigned short *gdata, unsigned short *odata, int *numChanges)
{
	__shared__ volatile unsigned int data[64][2][64];
	__shared__ volatile int cmem[32];

	// Calculate Indices
	unsigned int height = ((gridDim.y + 1) << 1);
	unsigned int width = ((gridDim.x + 1) << 5);
	int iz = ((blockIdx.z << 5) + threadIdx.z);
	int iy = (blockIdx.y << 1);
	int ix = ((blockIdx.x << 5) + threadIdx.x);
	int lidx = (iz * height + iy) * width + ix;

	// Load Data
	int fs = width * height * blockDim.z;
	unsigned int val[8];
	val[0] = gdata[lidx] | (gdata[lidx + width] << 16);
	val[1] = gdata[lidx + blockDim.x] | (gdata[lidx + blockDim.x + width] << 16);
	val[2] = gdata[lidx + 2 * width] | (gdata[lidx + 3 * width] << 16);
	val[3] = gdata[lidx + blockDim.x + 2 * width] | (gdata[lidx + blockDim.x + 3 * width] << 16);
	val[4] = gdata[fs + lidx] | (gdata[fs + lidx + width] << 16);
	val[5] = gdata[fs + lidx + blockDim.x] | (gdata[fs + lidx + blockDim.x + width] << 16);
	val[6] = gdata[fs + lidx + 2 * width] | (gdata[fs + lidx + 3 * width] << 16);
	val[7] = gdata[fs + lidx + blockDim.x + 2 * width] | (gdata[fs + lidx + blockDim.x + 3 * width] << 16);
	unsigned int oval[8];
#pragma unroll
	for (int i = 0; i < 8; i++) oval[i] = val[i];

	WriteShareData(data, val);
	__syncthreads();

	// Start iterative thinning
	unsigned int count = 1, iteration = 0;
	unsigned int n[26];

	while ((count > 0) && (iteration < 16)) // 16
	{
		iteration++;
		count = 0;

		// Calculate First Round Without Deleting
		unsigned int pv[8];
		PreTest<0>(pv, val, data, n);
		PreTest<1>(pv, val, data, n);
		PreTest<2>(pv, val, data, n);
		PreTest<3>(pv, val, data, n);
		PreTest<4>(pv, val, data, n);
		PreTest<5>(pv, val, data, n);
		PreTest<6>(pv, val, data, n);
		PreTest<7>(pv, val, data, n);

		// Eliminate Up Boundary Points
		GENERATE_BLOCK_MASK(Up)
		WriteShareData(data, val);
		__syncthreads();

		// Eliminate North Boundary Points
		GENERATE_BLOCK_MASK(North)
		WriteShareData(data, val);
		__syncthreads();

		// Eliminate East Boundary Points
		GENERATE_BLOCK_MASK(East)
		WriteShareData(data, val);
		__syncthreads();

		// Eliminate Down Boundary Points
		GENERATE_BLOCK_MASK(Down)
		WriteShareData(data, val);
		__syncthreads();

		// Eliminate South Boundary Points
		GENERATE_BLOCK_MASK(South)
		WriteShareData(data, val);
		__syncthreads();

		// Eliminate West Boundary Points
		GENERATE_BLOCK_MASK(West)
		WriteShareData(data, val);


		// Calculate Number of Overall Changes
#ifdef _USE_MASKED_COUNT_
		if (iteration == 16)
		{
			count = 0;
			if ((blockIdx.z == 0) || (threadIdx.z >= 16))
			{
				if ((blockIdx.x == 0) || (threadIdx.x >= 16))
				{
					count |= ((blockIdx.y > 0) ? ((val[0] ^ oval[0]) & 0xffff0000) : (val[0] ^ oval[0]));
					count |= ((blockIdx.y < (gridDim.y - 1)) ? ((val[2] ^ oval[2]) & 0x0000ffff) : (val[2] ^ oval[2]));
						
				}
				if ((blockIdx.x == (gridDim.x - 1)) || (threadIdx.x < 16))
				{
					count |= ((blockIdx.y > 0) ? ((val[1] ^ oval[1]) & 0xffff0000) : (val[1] ^ oval[1]));
					count |= ((blockIdx.y < (gridDim.y - 1)) ? ((val[3] ^ oval[3]) & 0x0000ffff) : (val[3] ^ oval[3]));
				}
			}
			if ((blockIdx.z == (gridDim.z - 1)) && (threadIdx.z < 16))
			{
				if ((blockIdx.x == 0) || (threadIdx.x >= 16))
				{
					count |= ((blockIdx.y > 0) ? ((val[4] ^ oval[4]) & 0xffff0000) : (val[4] ^ oval[4]));
					count |= ((blockIdx.y < (gridDim.y - 1)) ? ((val[6] ^ oval[6]) & 0x0000ffff) : (val[6] ^ oval[6]));
				}
				if ((blockIdx.x == (gridDim.x - 1)) || (threadIdx.x < 16))
				{
					count |= ((blockIdx.y > 0) ? ((val[5] ^ oval[5]) & 0xffff0000) : (val[5] ^ oval[5]));
					count |= ((blockIdx.y < (gridDim.y - 1)) ? ((val[7] ^ oval[7]) & 0x0000ffff) : (val[7] ^ oval[7]));
				}
			}
		}
#endif
		count = min(1,count);


		count = warpReduce3DLinear(count);
		if (threadIdx.x == 0) cmem[threadIdx.z] = count;

		__syncthreads();
		if (threadIdx.z == 0)
		{
			count = cmem[threadIdx.x];
			count = warpReduce3DLinear(count);
			cmem[threadIdx.x] = count;
		}
		__syncthreads();
		count = cmem[threadIdx.x];

	} // End Iterative Thinning

	// Write out results
	__syncthreads();
	if ((blockIdx.z == 0) || (threadIdx.z >= 16))
	{
		if ((blockIdx.x == 0) || (threadIdx.x >= 16))
		{
						
			if (blockIdx.y == 0)	odata[lidx] = (unsigned short)val[0];
			if (blockIdx.y == (gridDim.y - 1))	odata[lidx + 3 * width] = (unsigned short)(val[2] >> 16);

			odata[lidx + width] = (unsigned short)(val[0] >> 16);
			odata[lidx + 2 * width] = (unsigned short)val[2];
		}

		if ((blockIdx.x == (gridDim.x-1)) || (threadIdx.x < 16))
		{
							
			if (blockIdx.y == 0) odata[lidx + blockDim.x] = (unsigned short)val[1];
			if (blockIdx.y == (gridDim.y - 1))	odata[lidx + blockDim.x + 3 * width] = (unsigned short)(val[3] >> 16);

			odata[lidx + blockDim.x + width] = (unsigned short)(val[1] >> 16);
			odata[lidx + blockDim.x + 2 * width] = (unsigned short)val[3];
		}
	}

	if ((blockIdx.z == (gridDim.z - 1)) || (threadIdx.z < 16))
	{
		if ((blockIdx.x == 0) || (threadIdx.x >= 16))
		{
			
			if (blockIdx.y == 0) odata[fs + lidx] = (unsigned short)val[4];
			if (blockIdx.y == (gridDim.y - 1))	odata[fs + lidx + 3 * width] = (unsigned short)(val[6] >> 16);

			odata[fs + lidx + width] = (unsigned short)(val[4] >> 16);			
			odata[fs + lidx + 2 * width] = (unsigned short)val[6];
		}
		if ((blockIdx.x == (gridDim.x - 1)) || (threadIdx.x < 16))
		{
			
			if (blockIdx.y == 0) odata[fs + lidx + blockDim.x] = (unsigned short)val[5];
			if (blockIdx.y == (gridDim.y - 1))	odata[fs + lidx + blockDim.x + 3 * width] = (unsigned short)(val[7] >> 16);

			odata[fs + lidx + blockDim.x + 2 * width] = (unsigned short)val[7];
			odata[fs + lidx + blockDim.x + width] = (unsigned short)(val[5] >> 16);
		}
	}

	// Write num Changes
	if ((threadIdx.x == 0) && (threadIdx.z == 0))
	{
		numChanges[(gridDim.y*blockIdx.z + blockIdx.y)*gridDim.x + blockIdx.x] = count;
	}


}

#endif