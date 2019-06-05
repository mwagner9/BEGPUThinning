/**
 * @file Structs.h
 *
 * @brief Defines simple structures to define 2D points and curvilinear segments
 *
 * @author Martin Wagner
 * Contact: mwagner9@wisc.edu
 *
 */
 
#pragma once

#include <vector>

struct point2D
{
	short x;
	short y;
};

struct point2Df
{
	float x;
	float y;
};

struct SSegment
{
	std::vector<point2D> points;
	bool pruned;
	bool is_end_segment;

	SSegment() : pruned(false), is_end_segment(false) {}
};

