% MATLAB2DTHINNING This mex function performs thinning and centerline
% extraction of 3D binary volumes using a real-time GPU implementation as
% described in "Real-time thinning algorithms for 2D and 3D images using
% GPU processors" (Wagner, 2019, Journal of Real-Time Image Processing). 
%
%  USAGE: segs = Matlab3DThinning(vol, pruning_length, smoothing_span)
%
%  OUTPUT
%	segs:  Cell array of curvilinear segments. Each element of the cell
%          array is a Nx3 array containing 3D coordinates of a centerline
%          segment. Coordinates are zeros based.
%
% __________________________________________________________________________
%  INPUTS
%	img:  Binary image can be double, single, integer or logical format.
%         Pixels larger or equal to 1 are considered object pixels.
%   pruning_length: Segments with this length or smaller which are not
%                   connected on at least one end are removed. (default =
%                   0)
%   smoothing_span: Performs a moving average filter with the defined span
%                   on all centerline segments. (default = 0)
%
% __________________________________________________________________________
%  VARARGIN
%	See 'Parameter Initialization' section ...
%

% ------------------------------- Version 1.0 -------------------------------
%	Author:  Martin Wagner
%	Email:     mwagner9@wisc.edu
%	Created:  2019-06-05
% __________________________________________________________________________
