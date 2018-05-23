# pragma once

# include <string>
# include <map>
# include <iostream>
# include <vector>
# include <cmath>

# include "Partio.h"

# include "cuda_runtime.h"
# include "helper_functions.h"
# include "device_launch_parameters.h"
# include "helper_cuda.h"
# include "cufft.h"
//# include <Eigen/Eigen>

//# define OMParallelize
# ifdef OMParallelize
# include <omp.h>
# define TOTALThreads 16
# endif

# define M_PI           3.14159265358979323846  /* pi */
# define M_2PI			6.28318530717958647692  /* 2pi */
# define M_hPI			1.57079632679489661923  /* pi / 2*/

# define getIndex(phi, theta) (phi * nTheta + theta)

# define DEBUGBUILD

// The solution to switch between double and float
typedef float fReal;
typedef cufftDoubleComplex ComplexFourier;

typedef texture<float, 2, cudaReadModeElementType> table2D;

const size_t byte2Bits = 8;

const fReal density = 1000.0;
const fReal uSolid = 0.0;
const fReal vSolid = 0.0;

enum gridType { FLUIDGRID, SOLIDGRID };

enum Coord { phi, theta };

bool validatePhiTheta(fReal& phi, fReal& theta);