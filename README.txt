This code provides implementation of the real-time thinning / centerline extraction techniques
proposed in "Real-time thinning algorithms for 2D and 3D images using GPU processors" (Wagner, 
2019, Journal of Real-Time Image Processing). The library take binarized 2D images or 3D volumes
and generates a list of curvilinear segments representing the centerlines. The code is free to 
use for research purposes and non-commercial use. If you do so please cite:

Wagner, M.G. J Real-Time Image Proc (2019). https://doi.org/10.1007/s11554-019-00886-7

The paper can be found at https://link.springer.com/article/10.1007/s11554-019-00886-7.

If you have any issues using the code please contact me at mwagner9@wisc.edu.



INSTALLATION
=============

The code was written for Windows and tested on Matlab 2018b with CUDA toolkit version 9.1. It requires
a CUDA capable GPU with compute capability 3.x or higher. Other versions of Matlab and CUDA should work as well. 
Currently mexcuda does not support the new interleaved complex API, therefore, the separate complex API is used 
instead by default. If you have a newer version of MATLAB where the separate complex API is not supported anymore, 
please uncomment "#define USE_SEPARATE_COMPLEX_API" in both Matlab2DThinning.cpp and Matlab3DThinning.cpp.

1) Install CUDA toolkit: It is important to have the CUDA version corresponding to your Matlab version 
   installed. A list of the corresponding CUDA toolkit versions can be found here:
   https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html;jsessionid=7e482ca21ebcd63c2e1b53b0aa5a
   
2) (Optional) Set environment variable MW_NVCC_PATH to the CUDA path: This is required for mexcuda to work correctly.
   The provided Compile function will attempt to find the path automatically, however if you are using non-standard
   install paths for CUDA it might not be able to find it. You can set the variable from MATLAB using the following 
   command (Replace the path with your own installation path):
   setenv('MW_NVCC_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin');

3) Run Matlab function "Compile" in the main directory of the provided code. This will generate the mex files: 
   Matlab2DThinning.mexw64 and Matlab3DThinning.mexw64.
   
4) (Optional) Run the Examples code provided in Example2D.m and Example3D.m


COMPILE AS LIBRARY FOR C/C++
============================

This code can also be compiled as C/C++ library (.lib or .dll):

1) Create a new Project in Visual Studio and select the CUDA X.X Runtime project type.
2) Add all .h, .cpp, .cu, and .cuh files in the 2D or 3D folder to the project.
3) Go to project settings and set "Configuration Type" to "Dynamic Library (*.dll)" or "Static Library (*.lib)"
4) Build project

In your C/C++ project include header "CenterlineExtraction.h" and add lib file to linker.