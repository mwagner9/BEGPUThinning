/**
 * @file CUDA_SubiterationKernels3D.cuh
 *
 * @brief This files implements CUDA kernels to detect simple voxels in the six subiterations.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#ifndef _SUBITERATION_KERNELS_3D_CUH_
#define _SUBITERATION_KERNELS_3D_CUH_

__inline__ __device__ unsigned int Up3D(unsigned int val, unsigned int *n, unsigned int prev = 0)
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

	return val & ~(combined & ~prev);

} // End Up3D

__inline__ __device__ unsigned int Down3D(unsigned int val, unsigned int *n, unsigned int prev = 0)
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

	return val & ~(combined & ~prev);

} // End Down3D

__inline__ __device__ unsigned int South3D(unsigned int val, unsigned int *n, unsigned int prev = 0)
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

	return val & ~(combined & ~prev);

} // End South3D

__inline__ __device__ unsigned int North3D(unsigned int val, unsigned int *n, unsigned int prev = 0)
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

	return val & ~(combined & ~prev);

} // End North3D

__inline__ __device__ unsigned int East3D(unsigned int val, unsigned int *n, unsigned int prev = 0)
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

	return val & ~(combined & ~prev);

} // End East3D

__inline__ __device__ unsigned int West3D(unsigned int val, unsigned int *n, unsigned int prev = 0)
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

	return val & ~(combined & ~prev);

} // End West3D


#endif // !_SUBITERATION_KERNELS_3D_CUH_
