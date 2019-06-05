/**
 * @file ExtractSegments.h
 *
 * @brief This files declares functions to extract curvilinear segments from thinned binary volumes.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */


#pragma once

#include <vector>
#include "Structs.h"

// Does not allow object pixels along the border of the volume
template <typename PixelType>
std::vector<std::vector<point3D>> ExtractSegments(PixelType *data, int width, int height, int depth);

template <typename PixelType>
std::vector<SSegment *> ExtractSegmentsV(PixelType *data, int width, int height, int depth);
