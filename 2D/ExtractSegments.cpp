/**
 * @file ExtractSegments.cpp
 *
 * @brief This files implements functions to extract curvilinear segments from thinned binary images.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */
 
#include "ExtractSegments.h"


using namespace std;

template <typename PixelType>
vector<SSegment *> ExtractSegmentsV(PixelType *data, int width, int height)
{

	// Initialization
	vector<point2D> points;
	size_t nelem = width * height;
	PixelType *pdata = data;
	char *neighbors = new char[nelem];
	memset(neighbors, 0, nelem * sizeof(char));
	vector<SSegment *> segments;

	// Create Delta Array
	const int delta[8] = { -width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1 };

	// Extract Points
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++, pdata++)
		{
			if ((pdata[0] == 0) || (x == 0) || (y == 0) || (x == (width - 1)) || (y == (height - 1))) continue;
			int nn = 0;

			for (int ni = 0; ni < 8; ni++)
			{
				if (pdata[delta[ni]]) nn++;
			}

			if (nn == 0) nn = 1;
			if (nn != 2) nn = -nn;
			neighbors[y*width + x] = nn;
			point2D p = { x, y };
			points.push_back(p);
		}
	}

	char *onn = new char[nelem];
	memcpy(onn, neighbors, sizeof(char) * nelem);

	// Extract Segments
	for (auto it = points.begin(); it != points.end(); it++)
	{
		// Check if endpoint
		int idx = it->y*width + it->x;
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
			point2D sp = { it->x, it->y };
			seg->points.push_back(sp);

			// Find next neighbor
			ni++;
			for (; ni < 8; ni++)
			{
				if (neighbors[idx + delta[ni]] != 0) break;
			}

			// Check if single point
			if (ni >= 8)
			{
				segments.push_back(seg);
				continue;
			}

			// Follow Segment
			int tidx = idx + delta[ni];
			while (neighbors[tidx] == 2)
			{
				neighbors[tidx] = 0;
				int tx = tidx % width;
				int ty = tidx / width;

				point2D tp = { tx, ty };
				seg->points.push_back(tp);
				for (int ni2 = 0; ni2 < 8; ni2++)
				{
					if (neighbors[tidx + delta[ni2]] != 0)
					{
						tidx += delta[ni2];
						break;
					}
				}
			}

			// Add Last Point and push segment
			int ex = tidx % width;
			int ey = tidx / width;

			point2D ep = { ex, ey };
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


// Explicit template instantiations
template vector<SSegment *> ExtractSegmentsV<bool>(bool *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<unsigned char>(unsigned char *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<char>(char *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<signed char>(signed char *data, int width, int height); 
template vector<SSegment *> ExtractSegmentsV<unsigned short>(unsigned short *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<short>(short *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<unsigned int>(unsigned int *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<int>(int *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<float>(float *data, int width, int height);
template vector<SSegment *> ExtractSegmentsV<double>(double *data, int width, int height);