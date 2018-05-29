# include "../include/KaminoGPU.h"

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
	// stores BGR pixel values for an image
	// all values initialized to WHITE
	size_t size = nTheta * 2 * nTheta;
	this->colorMap = new vec3[size];
	for (int i = 0; i < size; ++i) {
		colorMap[i] = vec3(128.0, 128.0, 128.0);
	}

# ifdef OMParallelize
	omp_set_num_threads(TOTALThreads);
	Eigen::setNbThreads(TOTALThreads);
# endif

}

Kamino::~Kamino()
{
}

void Kamino::run()
{
	KaminoSolver solver(nPhi, nTheta, radius, dt, A, B, C, D, E);
	
	solver.write_data_bgeo(gridPath, 0);

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
		solver.write_data_bgeo(gridPath, i);
	}
}