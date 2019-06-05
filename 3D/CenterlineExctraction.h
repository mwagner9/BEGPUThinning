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
* Extracts centerlines from a binary volume (data: width x height x depth), 
* prunes the centerline graph by removing segments smaller than max_segment_length. 
* The segments are smoothed if smoothing_span is larger than 0. The x, y and z coordinates of all
* points are returned in the linear arrays out_x, out_y and out_z. The start 
* indices of all segments is given in out_idx, where the last index is the total
* number of points (out_idx[num_segs]). The number of segments is returned by the
* function. Width and height have to be a multiple of 32. Data should not contain any 
* object pixels along the boundary of the volume. The memory for out_x, out_y, out_z and out_idx
* is allocated by the function. The caller is responsible for deleteting this memory!
*/
template< typename PixelType>
unsigned int CenterlineExtraction(PixelType *data, int width, int height, int depth,
	int max_segment_length, int smoothing_span,
	float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
