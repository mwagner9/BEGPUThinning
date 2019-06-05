/**
 * @file LibThinning3D.h
 *
 * @brief This files declares the thinning function for binary volumes
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */

#pragma once

// Width, Height and Depth have to be a multiple of 32, data and out are pointers to host memory
template <typename PixelType>
void Thinning3D(int width, int height, int depth, PixelType *data, PixelType *out);


