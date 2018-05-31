# include "../include/KaminoGPU.h"
# define WRITE_BGEO

Kamino::Kamino(fReal radius, size_t nTheta, fReal particleDensity,
	float dt, float DT, int frames,
	fReal A, int B, int C, int D, int E,
	std::string gridPath, std::string particlePath,
	std::string densityImage, std::string solidImage, std::string colorImage) :
	radius(radius), nTheta(nTheta), nPhi(2 * nTheta), gridLen(M_PI / nTheta),
	particleDensity(particleDensity),
	dt(dt), DT(DT), frames(frames),
	A(A), B(B), C(C), D(D), E(E),
	gridPath(gridPath), particlePath(particlePath),
	densityImage(densityImage), solidImage(solidImage), colorImage(colorImage)
{
	size_t constantHostIntVars[2];
	fReal constantHostRealVars[3];

	constantHostIntVars[nThetaIdx] = nTheta;
	constantHostIntVars[nPhiIdx] = nPhi;

	constantHostRealVars[radiusIdx] = radius;
	constantHostRealVars[timeStepIdx] = dt;
	constantHostRealVars[gridLenIdx] = gridLen;

	checkCudaErrors(cudaMemcpyToSymbol(constantIntVars, constantHostIntVars,
		2 * sizeof(size_t), 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(constantRealVars, constantHostRealVars,
		3 * sizeof(fReal), 0, cudaMemcpyHostToDevice));
}

Kamino::~Kamino()
{
}

void Kamino::run()
{
	KaminoSolver solver(nPhi, nTheta, radius, dt, A, B, C, D, E);
	
# ifdef WRITE_BGEO
	solver.write_data_bgeo(gridPath, 0);
# endif

	float T = 0.0;              // simulation time
	for (int i = 1; i <= frames; i++) 
	{
		while (T < i*DT) 
		{
			solver.stepForward(dt);
			T += dt;
		}
		solver.stepForward(dt + i*DT - T);
		T = i*DT;

		std::cout << "Frame " << i << " is ready" << std::endl;
# ifdef WRITE_BGEO
		solver.write_data_bgeo(gridPath, i);
# endif
	}
}