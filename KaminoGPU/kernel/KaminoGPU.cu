# include "../include/KaminoGPU.cuh"

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
{}

Kamino::~Kamino()
{}

void Kamino::run()
{
	KaminoSolver solver(nPhi, nTheta, radius, dt, A, B, C, D, E);
	solver.initDensityfromPic(densityImage);
	solver.initParticlesfromPic(colorImage, this->particleDensity);
	
# ifdef WRITE_BGEO
	solver.write_data_bgeo(gridPath, 0);
	solver.write_particles_bgeo(particlePath, 0);
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
		solver.write_particles_bgeo(particlePath, i);
# endif
	}
}