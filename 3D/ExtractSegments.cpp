/**
 * @file ExtractSegments.cpp
 *
 * @brief This files implements functions to extract curvilinear segments from thinned binary volumes.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#include "ExtractSegments.h"

using namespace std;

template <typename PixelType>
vector<vector<point3D>> ExtractSegments( PixelType *data, int width, int height, int depth)
{

	// Initialization
	vector<point3D> points;
	size_t nelem = width * height * depth;
	int sliceelems = width * height;
	PixelType *pdata = data;
	char *neighbors = new char[nelem];
	memset(neighbors, 0, nelem * sizeof(char));
	vector<vector<point3D>> segments;

	// Create Delta Array
	const int delta[26] = { -sliceelems - width - 1, -sliceelems - width, -sliceelems - width + 1, -sliceelems - 1, -sliceelems, -sliceelems + 1, -sliceelems + width - 1, -sliceelems + width, -sliceelems + width + 1,
		-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1,
		sliceelems - width - 1, sliceelems - width, sliceelems - width + 1, sliceelems - 1, sliceelems, sliceelems + 1, sliceelems + width - 1, sliceelems + width, sliceelems + width + 1 };

	// Extract Points
	for (int z = 0; z < depth; z++)
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++, pdata++)
	{
		if (!pdata[0] || (x == 0) || (y == 0) || (z == 0) || (x == (width - 1)) || (y == (height - 1)) || (z == (depth - 1))) continue;
		int nn = 0;

		for (int ni = 0; ni < 26; ni++)
		{
			if (pdata[delta[ni]]) nn++;
		}

		if (nn == 0) nn = 1;
		if (nn != 2) nn = -nn;
		neighbors[z*sliceelems + y*width + x] = nn;
		point3D p = { (float)x, (float)y, (float)z};
		points.push_back(p);
	}

	// Extract Segments
	for (auto it = points.begin(); it != points.end(); it++)
	{
		// Check if endpoint
		int idx = it->z*sliceelems + it->y*width + it->x;
		char n = neighbors[idx];
		if (n >= 0) continue;

		// Set num neighbors to zero
		neighbors[idx] = 0;

		// Extract segments
		int ni = -1;
		for (char si = 0; si > n; si--)
		{
			// Create Segment
			vector<point3D> seg;
			point3D sp = { (float)it->x, (float)it->y, (float)it->z };
			seg.push_back(sp);


			// Find next neighbor
			ni++;
			for (; ni < 26; ni++)
			{
				if (neighbors[idx + delta[ni]] != 0) break;
			}

			// Check if single point
			if (ni >= 26)
			{
				segments.push_back(seg);
				continue;
			}

			// Follow Segment
			int tidx = idx + delta[ni];
			while (neighbors[tidx] == 2)
			{
				neighbors[tidx] = 0;
				int tz = tidx / sliceelems;
				int tx = tidx - (tz *sliceelems);
				int ty = tx / width;
				tx -= (ty * width);

				point3D tp = { tx, ty, tz };
				seg.push_back(tp);
				for (int ni2 = 0; ni2 < 26; ni2++)
				{
					if (neighbors[tidx + delta[ni2]] != 0)
					{
						tidx += delta[ni2];
						break;
					}
				}
			}

			// Add Last Point and push segment
			int ez = tidx / sliceelems;
			int ex = tidx - (ez *sliceelems);
			int ey = ex / width;
			ex -= (ey * width);

			point3D ep = { ex, ey, ez };
			seg.push_back(ep);
			neighbors[tidx] += 1;
			segments.push_back(seg);
		}

	}

	// Clean Memory
	delete[] neighbors;

	return segments;
}

template vector<vector<point3D>> ExtractSegments<bool>(bool *data, int width, int height, int depth);
template vector<vector<point3D>> ExtractSegments<unsigned char>(unsigned char *data, int width, int height, int depth);
template vector<vector<point3D>> ExtractSegments<float>(float *data, int width, int height, int depth);


template <typename PixelType>
vector<SSegment *> ExtractSegmentsV(PixelType *data, int width, int height, int depth)
{

	// Initialization
	vector<point3D> points;
	size_t nelem = width * height * depth;
	int sliceelems = width * height;
	PixelType *pdata = data;
	char *neighbors = new char[nelem];	
	memset(neighbors, 0, nelem * sizeof(char));
	vector<SSegment *> segments;

	// Create Delta Array
	const int delta[26] = { -sliceelems - width - 1, -sliceelems - width, -sliceelems - width + 1, -sliceelems - 1, -sliceelems, -sliceelems + 1, -sliceelems + width - 1, -sliceelems + width, -sliceelems + width + 1,
		-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1,
		sliceelems - width - 1, sliceelems - width, sliceelems - width + 1, sliceelems - 1, sliceelems, sliceelems + 1, sliceelems + width - 1, sliceelems + width, sliceelems + width + 1 };

	// Extract Points
	for (int z = 0; z < depth; z++)
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++, pdata++)
			{
				if (!pdata[0] || (x == 0) || (y == 0) || (z == 0) || (x == (width - 1)) || (y == (height - 1)) || (z == (depth - 1))) continue;
				int nn = 0;

				for (int ni = 0; ni < 26; ni++)
				{
					if (pdata[delta[ni]]) nn++;
				}

				if (nn == 0) nn = 1;
				if (nn != 2) nn = -nn;
				neighbors[z*sliceelems + y*width + x] = nn;
				point3D p = { (float)x, (float)y, (float)z };
				points.push_back(p);
			}
	char *onn = new char[nelem];
	memcpy(onn, neighbors, sizeof(char) * nelem);

	// Extract Segments
	for (auto it = points.begin(); it != points.end(); it++)
	{
		// Check if endpoint
		int idx = it->z*sliceelems + it->y*width + it->x;
		char n = neighbors[idx];
		if (n >= 0) continue;

		// Set num neighbors to zero
		neighbors[idx] = 0;

		// Extract segments
		int ni = -1;
		for (char si = 0; si > n; si--)
		{
			// Create Segment
			SSegment *seg = new SSegment();
			if (abs(onn[idx]) < 2) seg->is_end_segment = true;
			point3D sp = { (float)it->x, (float)it->y, (float)it->z };
			seg->points.push_back(sp);

			// Find next neighbor
			ni++;
			for (; ni < 26; ni++)
			{
				if (neighbors[idx + delta[ni]] != 0) break;
			}

			// Check if single point
			if (ni >= 26)
			{
				segments.push_back(seg);
				continue;
			}

			// Follow Segment
			int tidx = idx + delta[ni];
			while (neighbors[tidx] == 2)
			{
				neighbors[tidx] = 0;
				int tz = tidx / sliceelems;
				int tx = tidx - (tz *sliceelems);
				int ty = tx / width;
				tx -= (ty * width);

				point3D tp = { tx, ty, tz };
				seg->points.push_back(tp);
				for (int ni2 = 0; ni2 < 26; ni2++)
				{
					if (neighbors[tidx + delta[ni2]] != 0)
					{
						tidx += delta[ni2];
						break;
					}
				}
			}

			// Add Last Point and push segment
			int ez = tidx / sliceelems;
			int ex = tidx - (ez *sliceelems);
			int ey = ex / width;
			ex -= (ey * width);

			point3D ep = { ex, ey, ez };
			seg->points.push_back(ep);
			if (abs(onn[tidx]) < 2) seg->is_end_segment = true;
			neighbors[tidx] += 1;
			segments.push_back(seg);
		}

	}

	// Clean Memory
	delete[] neighbors;

	return segments;
}

template vector<SSegment *> ExtractSegmentsV<bool>(bool *data, int width, int height, int depth);
template vector<SSegment *> ExtractSegmentsV<unsigned char>(unsigned char *data, int width, int height, int depth);
template vector<SSegment *> ExtractSegmentsV<float>(float *data, int width, int height, int depth);
