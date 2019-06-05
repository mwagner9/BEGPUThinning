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

#include "CenterlineExctraction.h"
#include "ExtractSegments.h"
#include <vector>
#include <limits>
#include "Structs.h"
#include <algorithm>
#include "LibThinning3D.h"

using namespace std;

template< typename PixelType>
unsigned int CenterlineExtraction(PixelType *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx)
{

	// Convert Volume
	int pwidth = int((width + 31) / 32) * 32;
	int pheight = int((height + 31) / 32) * 32;
	int pdepth = int((depth + 31) / 32) * 32;

	int nelems = pwidth*pheight*pdepth;
	bool *ivol = new bool[nelems];
	bool *ovol = new bool[nelems];

	bool *t_ptr = ivol;
	PixelType *s_ptr = data;
	for (int iz = 0; iz < depth; iz++)
	{
		for (int iy = 0; iy < height; iy++)
		{
			t_ptr = &ivol[(iz*pheight + iy)*pwidth];
			for (int ix = 0; ix < width; ix++, s_ptr++)
			{
				t_ptr[ix] = int(*s_ptr) > 0;
			}
		}
	}

	// Extract segments
	Thinning3D<bool>(pwidth, pheight, pdepth, ivol, ovol);
	vector<SSegment *> segs = ExtractSegmentsV<bool>(ovol, pwidth, pheight, pdepth);
	delete[] ivol;
	delete[] ovol;

	// Prune Tree
	for (auto it = segs.begin(); it != segs.end(); it++)
	{
		SSegment *s = *it;
		if (s->points.size() > max_segment_length) continue;
		if (!s->is_end_segment) continue;
		s->pruned = true;
	}

	// Smooth Segments
	vector<vector<point3Df>> segout;
	{
		for (auto it = segs.begin(); it != segs.end(); it++)
		{
			SSegment *s = *it;
			if (s->pruned) continue;
			vector<point3Df> pts;
			point3Df tpt = { float(s->points[0].x), float(s->points[0].y), float(s->points[0].z) };
			pts.push_back(tpt);
			float x = s->points[0].x, y = s->points[0].y, z = s->points[0].z;
			int span = min<int>(smoothing_span, int((s->points.size()-1)/2));
			int n = 1;

			for (int k = 1; k <= span; k++)
			{
				x += s->points[2 * k - 1].x;
				y += s->points[2 * k - 1].y;
				z += s->points[2 * k - 1].z;
				x += s->points[2 * k].x;
				y += s->points[2 * k].y;
				z += s->points[2 * k].z;
				n += 2;
				point3Df pt = { float(x) / float(n), float(y) / float(n), float(z) / float(n) };
				pts.push_back(pt);
			}
			for (int k = span + 1; k < s->points.size() - span; k++)
			{
				x += s->points[k + span].x;
				y += s->points[k + span].y;
				z += s->points[k + span].z;
				x -= s->points[k - span - 1].x;
				y -= s->points[k - span - 1].y;
				z -= s->points[k - span - 1].z;
				point3Df pt = { float(x) / float(n), float(y) / float(n), float(z) / float(n) };
				pts.push_back(pt);
			}
			for (int k = (s->points.size() - span), d = span; k < s->points.size(); k++, d--)
			{
				x -= s->points[k - d - 1].x;
				y -= s->points[k - d - 1].y;
				z -= s->points[k - d - 1].z;
				x -= s->points[k - d].x;
				y -= s->points[k - d].y;
				z -= s->points[k - d].z;
				n -= 2;
				point3Df pt = { float(x) / float(n), float(y) / float(n), float(z) / float(n) };
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
		npts += (*it).size();
	}

	// Allocating Output Memory
	*out_x = new float[npts];
	*out_y = new float[npts];
	*out_z = new float[npts];
	*out_idx = new unsigned int[nsegs+1];

	// Converting to Raw Data Pointers
	unsigned int idx = 0, pidx = 0;
	for (auto it = segout.begin(); it != segout.end(); it++)
	{
		(*out_idx)[idx++] = pidx;
		for (int i = 0; i < it->size(); i++, pidx++)
		{
			(*out_x)[pidx] = (*it)[i].x;
			(*out_y)[pidx] = (*it)[i].y;
			(*out_z)[pidx] = (*it)[i].z;
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

template __declspec(dllexport) unsigned int CenterlineExtraction<float>(float *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<double>(double *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<int>(int *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<unsigned int>(unsigned int *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<short>(short *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<unsigned short>(unsigned short *data, int width, int depth, int height, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<char>(char *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<signed char>(signed char *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<unsigned char>(unsigned char *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
template unsigned int CenterlineExtraction<bool>(bool *data, int width, int height, int depth, int max_segment_length, int smoothing_span, float **out_x, float **out_y, float **out_z, unsigned int **out_idx);
