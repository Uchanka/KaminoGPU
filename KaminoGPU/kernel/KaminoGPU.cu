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
	//solver.initParticlesfromPic(colorImage, this->particleDensity);
	
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	cudaEventRecord(start, 0);

# ifdef WRITE_BGEO
	solver.write_data_bgeo(gridPath, 0);
	//solver.write_particles_bgeo(particlePath, 0);
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
		//solver.write_particles_bgeo(particlePath, i);
# endif
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
	std::cout << "Time spent: " << gpu_time << "ms" << std::endl;
	std::cout << "Performance: " << 1000.0 * frames / gpu_time << " steps per second" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}