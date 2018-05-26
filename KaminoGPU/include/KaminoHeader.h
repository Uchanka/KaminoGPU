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
# include "vectorUtil.h"

//# include <Eigen/Eigen>

//# define OMParallelize
# ifdef OMParallelize
# include <omp.h>
# define TOTALThreads 16
# endif

# define M_PI           3.14159265358979323846  /* pi */
# define M_2PI			6.28318530717958647692  /* 2pi */
# define M_hPI			1.57079632679489661923  /* pi / 2*/

# define centeredPhiOffset 0.0
# define centeredThetaOffset 0.5
# define vPhiPhiNorm M_2PI;
# define vPhiThetaNorm M_PI;
# define vThetaPhiNorm M_2PI;
# define vThetaThetaNorm (M_PI - 2 * gridLen)
# define pressurePhiNorm M_2PI
# define pressureThetaNorm M_PI
# define vPhiPhiOffset -0.5
# define vPhiThetaOffset 0.5
# define vThetaPhiOffset 0.0
# define vThetaThetaOffset 1.0

# define getIndex(phi, theta) (theta * this->nPhi + phi)

# define DEBUGBUILD

// The solution to switch between double and float
typedef float fReal;
typedef cufftComplex ComplexFourier;

typedef texture<fReal, 2, cudaReadModeElementType> table2D;

const size_t byte2Bits = 8;

const fReal density = 1000.0;
const fReal uSolid = 0.0;
const fReal vSolid = 0.0;

enum gridType { FLUIDGRID, SOLIDGRID };

enum Coord { phi, theta };

bool validatePhiTheta(fReal& phi, fReal& theta);