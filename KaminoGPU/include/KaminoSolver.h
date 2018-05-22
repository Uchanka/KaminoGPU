# pragma once

# include "KaminoQuantity.h"

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
	
	KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal gridLength, fReal frameDuration,
		fReal A, int B, int C, int D, int E);
	~KaminoSolver();

	void stepForward(fReal timeStep);

	void addCenteredAttr(std::string name, fReal xOffset = 0.5, fReal yOffset = 0.5);
	void addStaggeredAttr(std::string name, fReal xOffset, fReal yOffset);

	KaminoQuantity* getAttributeNamed(std::string name);
	KaminoQuantity* operator[](std::string name);

	gridType* getGridTypeHandle();
	void write_data_bgeo(const std::string& s, const int frame);
};