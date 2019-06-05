/**
 * @file CenterlineExtraction.cpp
 *
 * @brief This files implements the main functions provided by the library, which calls thinning and segment extraction 
*		  and optionally performs segment smoothing and pruning.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */
 
#include "CenterlineExtraction.h"
#include "ExtractSegments.h"
#include <vector>
#include <limits>
#include "Structs.h"
#include <algorithm>
#include "LibThinning2D.h"

using namespace std;

template< typename PixelType>
unsigned int CenterlineExtraction(PixelType *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx)
{

	// Padding Image
	int pwidth = int((width + 31) / 32) * 32;
	int pheight = std::max<int>(1, int((height + 63) / 128)) * 128 + 64;
	int nelems = pwidth*pheight;

	PixelType *timg = new PixelType[nelems];
	memset(timg, 0, sizeof(PixelType)*nelems);
	PixelType *t_ptr = timg;
	PixelType *s_ptr = data;
	for (int hi = 0; hi < height; hi++, t_ptr += pwidth, s_ptr += width)
	{
		memcpy(t_ptr, s_ptr, width*sizeof(PixelType));
	}


	// Convert Volume
	PixelType *oimg = new PixelType[nelems];

	// Extract segments
	Thinning2D<PixelType>(pwidth, pheight, timg, oimg);
	vector<SSegment *> segs = ExtractSegmentsV<PixelType>(oimg, pwidth, pheight);
	delete[] oimg;
	delete[] timg;

	// Prune Tree
	for (auto it = segs.begin(); it != segs.end(); it++)
	{
		SSegment *s = *it;
		if (s->points.size() > max_segment_length) continue;
		if (!s->is_end_segment) continue;
		s->pruned = true;
	}

	// Smooth Segments
	vector<vector<point2Df>> segout;
	{
		for (auto it = segs.begin(); it != segs.end(); it++)
		{
			SSegment *s = *it;
			if (s->pruned) continue;
			vector<point2Df> pts;
			point2Df tpt = { float(s->points[0].x), float(s->points[0].y) };
			pts.push_back(tpt);
			float x = s->points[0].x, y = s->points[0].y;
			int span = min<int>(smoothing_span, int((s->points.size() - 1) / 2));
			int n = 1;

			for (int k = 1; k <= span; k++)
			{
				x += s->points[2 * k - 1].x;
				y += s->points[2 * k - 1].y;
				x += s->points[2 * k].x;
				y += s->points[2 * k].y;
				n += 2;
				point2Df pt = { float(x) / float(n), float(y) / float(n)};
				pts.push_back(pt);
			}
			for (int k = span + 1; k < s->points.size() - span; k++)
			{
				x += s->points[k + span].x;
				y += s->points[k + span].y;
				x -= s->points[k - span - 1].x;
				y -= s->points[k - span - 1].y;
				point2Df pt = { float(x) / float(n), float(y) / float(n) };
				pts.push_back(pt);
			}
			for (int k = int(s->points.size() - span), d = span; k < s->points.size(); k++, d--)
			{
				x -= s->points[k - d - 1].x;
				y -= s->points[k - d - 1].y;
				x -= s->points[k - d].x;
				y -= s->points[k - d].y;
				n -= 2;
				point2Df pt = { float(x) / float(n), float(y) / float(n) };
				pts.push_back(pt);
			}
			segout.push_back(pts);
		}
	}

	// Count Number of Points and Segments
	int npts = 0, nsegs = 0;
	for (auto it = segout.begin(); it != segout.end(); it++)
	{
		//if ((*it)->pruned) continue;
		nsegs++;
		npts += int((*it).size());
	}

	// Allocating Output Memory
	*out_x = new float[npts];
	*out_y = new float[npts];
	*out_idx = new unsigned int[nsegs + 1];

	// Converting to Raw Data Pointers
	unsigned int idx = 0, pidx = 0;
	for (auto it = segout.begin(); it != segout.end(); it++)
	{
		(*out_idx)[idx++] = pidx;
		for (int i = 0; i < it->size(); i++, pidx++)
		{
			(*out_x)[pidx] = (*it)[i].x;
			(*out_y)[pidx] = (*it)[i].y;
		}
	}
	(*out_idx)[nsegs] = npts;

	// Clean up Segments
	for (auto it = segs.begin(); it != segs.end(); it++)
	{
		SSegment *s = *it;
		delete s;
	}

	return nsegs;

}


// Explicit Template Instantiations
template unsigned int CenterlineExtraction<float>(float *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<double>(double *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<int>(int *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<unsigned int>(unsigned int *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<short>(short *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<unsigned short>(unsigned short *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<char>(char *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<signed char>(signed char *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<unsigned char>(unsigned char *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
template unsigned int CenterlineExtraction<bool>(bool *data, int width, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, unsigned int **out_idx);
