/**
 * @file Structs.h
 *
 * @brief Defines simple structures to define 3D points and curvilinear segments
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */
 
#pragma once

#include <vector>

struct point3D
{
	short x;
	short y;
	short z;
};

struct point3Df
{
	float x;
	float y;
	float z;
};

struct SSegment
{
	std::vector<point3D> points;
	bool pruned;
	bool is_end_segment;

	SSegment() : pruned(false), is_end_segment(false) {}
};

