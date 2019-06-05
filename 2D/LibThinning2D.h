/**
 * @file LibThinning2D.h
 *
 * @brief This files declares the thinning function for binary images.
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */
 
#pragma once

// Width and height have to be a multiple of 32, data and out are pointers to host memory
template <typename PixelType>
void Thinning2D(int width, int height, PixelType *data, PixelType *out);


