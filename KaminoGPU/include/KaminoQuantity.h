# pragma once

# include <string>
# include <map>
# include <iostream>
# include <vector>

//# ifndef _MSC_VER
# include "Partio.h"
//# endif
# include <cmath>

//# define OMParallelize

# ifdef OMParallelize
# include <omp.h>
# define TOTALThreads 16
# endif

# define M_PI           3.14159265358979323846  /* pi */
# define M_2PI			6.28318530717958647692  /* 2pi */
# define M_hPI			1.57079632679489661923  /* pi / 2*/

# define DEBUGBUILD

# include <Eigen/IterativeLinearSolvers>
# include <unsupported/Eigen/IterativeSolvers>
# include <unsupported/Eigen/FFT>

// The solution to switch between double and float
typedef double fReal;

const fReal density = 1000.0;
const fReal uSolid = 0.0;
const fReal vSolid = 0.0;

enum gridType { FLUIDGRID, SOLIDGRID };

enum Coord { phi, theta };

class KaminoQuantity
{
private:
	/* Name of the attribute */
	std::string attrName;

	/* Grid dimensions */
	size_t nPhi;
	size_t nTheta;

	/* Grid size */
	fReal gridLen;
	/* 1.0 / gridlen */
	fReal invGridLen;

	/* Staggered? */
	fReal xOffset;
	fReal yOffset;

	/* Initial buffer at client side */
	fReal* cpuBuffer;
	/* Double buffer at server side */
	fReal* gpuThisStep;
	fReal* gpuNextStep;

	/* Get index */
	size_t getIndex(size_t phi, size_t theta);

public:
	/* Constructor */
	KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
		fReal gridLen, fReal xOffset, fReal yOffset);
	/* Destructor */
	~KaminoQuantity();

	/* Swap the buffer */
	void swapBuffer();
	/* Get its name */
	std::string getName();
	/* Get nx */
	size_t getNPhi();
	/* Get ny */
	size_t getNTheta();
	/* Get the current step */
	fReal getValueAt(size_t x, size_t y);
	/* Set the current step */
	void setValueAt(size_t x, size_t y, fReal val);
	/* Write to the next step */
	void writeValueTo(size_t x, size_t y, fReal val);
	/* Access */
	fReal& accessValueAt(size_t x, size_t y);
	/* Lerped Sampler using world coordinates */
	fReal sampleAt(fReal x, fReal y, fReal uNorth[2], fReal uSouth[2]);
	/* Given the index, show its origin in world coordinates*/
	fReal getPhiCoordAtIndex(size_t phi);
	fReal getThetaCoordAtIndex(size_t theta);
	/* And given world coordinates, show its index backwards... */
	size_t getPhiIndexAtCoord(fReal phi);
	size_t getThetaIndexAtCoord(fReal theta);

	fReal getPhiOffset();
	fReal getThetaOffset();
};

bool validatePhiTheta(fReal& phi, fReal& theta);

class KaminoSolver
{
private:
	// Buffer for the capital U
	fReal* gpuFourierUReal;
	fReal* gpuFourierUImag;
	// Buffer for the divergence, before the transform
	fReal* gpuBefFourierF;

	// Buffer for the divergence, F n theta.
	fReal* gpuFourieredFReal;
	fReal* gpuFourieredFImag;
	// Diagonal elements a (lower);
	fReal* gpuA;
	// Diagonal elements b (major diagonal);
	fReal* gpuB;
	// Diagonal elements c (upper);
	fReal* gpuC;
	// Divergence fourier coefficients
	fReal* gpuDReal;
	fReal* gpuDImag;
	// GPU side implementation wouldn't be in-place so...
	fReal* gpuXReal;
	fReal* gpuXImag;

	/* Grid types */
	gridType* gridTypes;
	/* Grid dimensions */
	size_t nPhi;
	size_t nTheta;
	/* Radius of sphere */
	fReal radius;
	/* Grid size */
	fReal gridLen;
	/* Inverted grid size*/
	fReal invGridLen;

	/* harmonic coefficients for velocity field initializaton */
	fReal A;
	int B, C, D, E;

	/* So that it remembers all these attributes within */
	std::map<std::string, KaminoQuantity*> centeredAttr;
	std::map<std::string, KaminoQuantity*> staggeredAttr;

	/* Something about time steps */
	fReal frameDuration;
	fReal timeStep;
	fReal timeElapsed;

	void resetPoleVelocities();
	void averageVelocities();
	void solvePolarVelocities();

	// Is it solid? or fluid? or even air?
	gridType getGridTypeAt(size_t x, size_t y);

	// We only have to treat uTheta differently
	void advectAttrAt(KaminoQuantity* attr, size_t gridPhi, size_t gridTheta);

	void advectionScalar();
	void advectionSpeed();

	void geometric();
	void projection();
	void bodyForce();

	void fillDivergence();
	void transformDivergence();
	void invTransformPressure();

	// Swap all these buffers of the attributes.
	void swapAttrBuffers();

	/* distribute initial velocity values at grid points */
	void initialize_velocity();
	/* initialize pressure attribute */
	void initialize_pressure();
	/* initialize density distribution */
	void initialize_density();
	/* which grids are solid? */
	void initialize_boundary();
	/* sum of sine functions for velocity initialization */
	fReal fPhi(const fReal x);
	/* */
	fReal gTheta(const fReal y);
	/* */
	fReal lPhi(const fReal x);
	/* */
	fReal mTheta(const fReal y);
	/* FBM noise function for velocity distribution */
	fReal FBM(const fReal x, const fReal y);
	/* 2D noise interpolation function for smooth FBM noise */
	fReal interpNoise2D(const fReal x, const fReal y) const;
	
	/* Tri-diagonal matrix solver */
	void TDMSolve(fReal* a, fReal* b, fReal* c, fReal* d);

public:
	// Velocities at poles in xyz cartesian coordinates
	//fReal uThetaNorthP[2];
	fReal uNorthP[2];
	//fReal uThetaSouthP[2];
	fReal uSouthP[2];

	KaminoSolver(size_t nx, size_t ny, fReal radius, fReal gridLength, fReal frameDuration,
		fReal A, int B, int C, int D, int E);
	~KaminoSolver();

	void stepForward(fReal timeStep);

	void addCenteredAttr(std::string name, fReal xOffset = 0.5, fReal yOffset = 0.5);
	void addStaggeredAttr(std::string name, fReal xOffset, fReal yOffset);

	KaminoQuantity* getAttributeNamed(std::string name);
	KaminoQuantity* operator[](std::string name);

	gridType* getGridTypeHandle();
	void write_data_bgeo(const std::string& s, const int frame);

	/* Duplicate of quantity's get index */
	size_t getIndex(size_t x, size_t y);
};