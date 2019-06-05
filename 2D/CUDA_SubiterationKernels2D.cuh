/**
 * @file CUDA_SubiterationKernels2D.cuh
 *
 * @brief This files implements CUDA kernels to detect simple pixels in the four subiterations.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */
 
#ifndef CUDA_SUBITERATION_KERNELS_2D_CUH
#define CUDA_SUBITERATION_KERNELS_2D_CUH


__inline__ __device__ int North(unsigned int up, unsigned int down, unsigned int &cp)
{

	unsigned int y1, y2, y3, y4, y5, t1, t2;

	y3 = ShiftLeft(cp);		// x6
	t1 = ShiftRight(up);	// x1
	y2 = ShiftRight(cp);	// x2
	t2 = ShiftLeft(up);		// x7
	y5 = ShiftRight(down);	// x3
	y4 = ShiftLeft(down);	// x5

	y1 = ~t1 & (y2 ^ y5 ^ ~down) & (y5 ^ y4) & (y5 ^ y3) & ~t2; // y1
	y4 = (t1 ^ y4) & (y2 ^ y4) & (down ^ ~y4) & ~y3 & ~t2; // y4
	y5 = ~t1 & ~y2 & ~y5 & y3 & t2; // y5
	y2 &= (down & (y3 | ~t2)); // y2
	y3 &= (~t1 & down); // y3

	y1 |= (y2 | y3 | y4 | y5);
	y1 &= ~up;

	y2 = y1 & cp;
	cp &= ~y1;

	return min(1, y2);
}

__device__ int East(unsigned int up, unsigned int down, unsigned int &cp)
{

	unsigned int y1, y2, y3, y4, y5, t1;

	y3 = ShiftLeft(cp);		// x6
	t1 = ShiftRight(down);	// x3
	y2 = ShiftRight(up);	// x1
	y5 = ShiftLeft(down);	// x5
	y4 = ShiftLeft(up);		// x7

	y1 = ~t1 & (down ^ y5 ^ ~y3) & (y5 ^ y4) & (y5 ^ up) & ~y2; // y1 
	y4 = (t1 ^ y4) & (down ^ y4) & (y3 ^ ~y4) & ~up & ~y2; // y4 
	y5 = ~t1 & ~down & ~y5 & up & y2; // y5 
	y2 = (down & y3 & (up | ~y2)); // y2
	y3 &= (~t1 & up); // y3

	y1 |= (y2 | y3 | y4 | y5);
	y1 &= ~ShiftRight(cp);

	y2 = y1 & cp;
	cp &= ~y1;

	return min(1, y2);
}

__device__ int South(unsigned int up, unsigned int down, unsigned int &cp)
{

	unsigned int y1, y2, y3, y4, y5, t1, t2;

	y3 = ShiftRight(cp);	// x2
	t1 = ShiftLeft(down);	// x5
	y2 = ShiftLeft(cp);		// x6
	t2 = ShiftRight(down);	// x3
	y5 = ShiftLeft(up);		// x7
	y4 = ShiftRight(up);	// x1

	y1 = ~t1 & (y2 ^ y5 ^ ~up) & (y5 ^ y4) & (y5 ^ y3) & ~t2; // y1 
	y4 = (t1 ^ y4) & (y2 ^ y4) & (up ^ ~y4) & ~y3 & ~t2; // y4 
	y5 = ~t1 & ~y2 & ~y5 & y3 & t2; // y5
	y2 &= (up & (y3 | ~t2)); // y2
	y3 &= (~t1 & up); // y3 !x5*x0*x2

	y1 |= (y2 | y3 | y4 | y5);
	y1 &= ~down;

	y2 = y1 & cp;
	cp &= ~y1;

	return min(1, y2);
}

__device__ int West(unsigned int up, unsigned int down, unsigned int &cp)
{

	unsigned int y1, y2, y3, y4, y5, t1;

	y3 = ShiftRight(cp);	// x2
	t1 = ShiftLeft(up);		// x7
	y2 = ShiftLeft(down);	// x5
	y5 = ShiftRight(up);	// x1
	y4 = ShiftRight(down);	// x3

	y1 = ~t1 & (up ^ y5 ^ ~y3) & (y5 ^ y4) & (y5 ^ down) & ~y2; // y1 !x7*(x0 o x1 o !x2)*(x1 o x3)*(x1 o x4)*!x5
	y4 = (t1 ^ y4) & (up ^ y4) & (y3 ^ ~y4) & ~down & ~y2; // y4  (x7 o x3)*(x0 o x3)*(x2 o !x3)*!x4*!x5
	y5 = ~t1 & ~up & ~y5 & down & y2; // y5  !x7*!x0*!x1*x4*x5
	y2 = (up & y3 & (down | ~y2)); // y2 x0*x2*(x4 | !x5)
	y3 &= (~t1 & down); // y3 !x7*x2*x4

	y1 |= (y2 | y3 | y4 | y5);
	y1 &= ~ShiftLeft(cp);

	y2 = y1 & cp;
	cp &= ~y1;

	return min(1, y2);
}


#endif // !CUDA_SUBITERATION_KERNELS_2D_CUH
