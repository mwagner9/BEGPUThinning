/**
 * @file ExtractSegments.h
 *
 * @brief This files declares functions to extract curvilinear segments from thinned binary images.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#pragma once

#include <vector>
#include "Structs.h"

template <typename PixelType>
std::vector<SSegment *> ExtractSegmentsV(PixelType *data, int width, int height);
