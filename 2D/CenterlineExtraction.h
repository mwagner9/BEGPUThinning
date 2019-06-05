/**
 * @file CenterlineExtraction.h
 *
 * @brief This files declares the main function provided by the library, which calls thinning and segment extraction 
*		  and optionally performs segment smoothing and pruning.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#pragma once

/******************************************************************************
* Extracts centerlines from a binary image (data: width x height),
* prunes the centerline graph by removing segments smaller than max_segment_length.
* The segments are smoothed if smoothing_span is larger than 0. The x and y coordinates of all
* points are returned in the linear arrays out_x and out_y. The start
* indices of all segments is given in out_idx, where the last index is the total
* number of points (out_idx[num_segs]). The number of segments is returned by the
* function. Data should not contain any object pixels along the boundary of the image. 
* The memory for out_x, out_y, and out_idx is allocated by the function. 
* The caller is responsible for deleteting this memory!
*/
template< typename PixelType> 
unsigned int CenterlineExtraction(PixelType *data, int width, int height,
	int max_segment_length, int smoothing_span,
	float **out_x, float **out_y, unsigned int **out_idx);
