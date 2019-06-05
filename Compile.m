function Compile(is_debug)
% COMPILE Compiles the Matlab version of the 2D and 3D centerline extration algorithms as
% described in "Real-time thinning algorithms for 2D and 3D images using GPU processors" 
% (Wagner 2019, Journal of Real-Time Image Processing).
%
%  USAGE: Compile(is_debug)
%
%  INPUTS
%	is_debug (optional):  Set this to true to compile in debug mode.
%                         Default is false.
%


% ------------------------------- Version 1.0 -------------------------------
%	Author:  Martin Wagner
%	Email:     mwagner9@wisc.edu
%	Created:  2019-06-04
% __________________________________________________________________________

%% Set Optional Parameters
if nargin < 1; is_debug = false; end

%% Setup compiler
if isempty(getenv('MW_NVCC_PATH'))
    warning('MW_NVCC_PATH has not been set. Trying to find CUDA path automatically ...\n');
    v = version('-release');
    cv = '';
    if strcmpi(v, '2019a'); cv = 'v10.0';
    elseif strcmpi(v, '2018b'); cv = 'v9.1';
    elseif strcmpi(v, '2018a'); cv = 'v9.0';
    elseif strcmpi(v, '2017b'); cv = 'v8.0';
    elseif strcmpi(v, '2017a'); cv = 'v8.0';
    elseif strcmpi(v, '2016b'); cv = 'v7.5';
    elseif strcmpi(v, '2016a'); cv = 'v7.5';
    elseif strcmpi(v, '2015b'); cv = 'v7.0';
    elseif strcmpi(v, '2015a'); cv = 'v6.5';
    elseif strcmpi(v, '2014b'); cv = 'v6.0';
    elseif strcmpi(v, '2014a'); cv = 'v5.5';
    end
    if isempty(cv); error('Unable to identify correct CUDA Toolkit version. Please set MW_NVCC_PATH to the correct directory'); end
    
    defcudadir = fullfile('C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA', cv, 'bin');
    if ~exist(defcudadir, 'dir')
        error('Unable to find CUDA Toolkit version %s in the default directory. Please install toolkit or manually set MW_NVCC_PATH');
    end
    setenv('MW_NVCC_PATH', defcudadir);
end
    
%% Clear from Memory
clear Matlab2DThinning;
clear Matlab3DThinning;

%% Compile
if is_debug
    mexcuda -G -g 3D/Matlab3DThinning.cpp 3D/LibThinning3D.cu 3D/ExtractSegments.cpp 3D/CenterlineExtraction.cpp
    mexcuda -G -g 2D/Matlab2DThinning.cpp 2D/LibThinning2D.cu 2D/ExtractSegments.cpp 2D/CenterlineExtraction.cpp
else
    mexcuda 3D/Matlab3DThinning.cpp 3D/LibThinning3D.cu 3D/ExtractSegments.cpp 3D/CenterlineExtraction.cpp
    mexcuda 2D/Matlab2DThinning.cpp 2D/LibThinning2D.cu 2D/ExtractSegments.cpp 2D/CenterlineExtraction.cpp
end
