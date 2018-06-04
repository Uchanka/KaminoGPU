# include "../include/KaminoSolver.cuh"
# include "../opencv_headers/opencv2/opencv.hpp"

// CONSTRUCTOR / DESTRUCTOR >>>>>>>>>>

const int fftRank = 1;
static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal gridLenGlobal;

KaminoSolver::KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal frameDuration,
	fReal A, int B, int C, int D, int E) :
	nPhi(nPhi), nTheta(nTheta), radius(radius), gridLen(M_2PI / nPhi), invGridLen(1.0 / gridLen), frameDuration(frameDuration),
	timeStep(0.0), timeElapsed(0.0),
	A(A), B(B), C(C), D(D), E(E)
{
	/// FIXME: Should we detect and use device 0?
	/// Replace it later with functions from helper_cuda.h!
	checkCudaErrors(cudaSetDevice(0));

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	this->nThreadxMax = deviceProp.maxThreadsDim[0];

	checkCudaErrors(cudaMalloc((void **)&gpuUFourier,
		sizeof(ComplexFourier) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)&gpuUReal,
		sizeof(fReal) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)&gpuUImag,
		sizeof(fReal) * nPhi * nTheta));

	checkCudaErrors(cudaMalloc((void **)&gpuFFourier,
		sizeof(ComplexFourier) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)&gpuFReal,
		sizeof(fReal) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)&gpuFImag,
		sizeof(fReal) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void**)&gpuFZeroComponent,
		sizeof(fReal) * nTheta));

	checkCudaErrors(cudaMalloc((void **)(&gpuA),
		sizeof(fReal) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)(&gpuB),
		sizeof(fReal) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)(&gpuC),
		sizeof(fReal) * nPhi * nTheta));
	precomputeABCCoef();

	this->velPhi = new KaminoQuantity("velPhi", nPhi, nTheta,
		vPhiPhiOffset, vPhiThetaOffset);
	this->velTheta = new KaminoQuantity("velTheta", nPhi, nTheta - 1,
		vThetaPhiOffset, vThetaThetaOffset);
	this->pressure = new KaminoQuantity("p", nPhi, nTheta,
		centeredPhiOffset, centeredThetaOffset);
	this->density = new KaminoQuantity("density", nPhi, nTheta,
		centeredPhiOffset, centeredThetaOffset);

	initialize_velocity();

	int sigLenArr[1];
	sigLenArr[0] = nPhi;
	checkCudaErrors((cudaError_t)cufftPlanMany(&kaminoPlan, fftRank, sigLenArr,
		NULL, 1, nPhi,
		NULL, 1, nPhi,
		CUFFT_C2C, nTheta));
}

KaminoSolver::~KaminoSolver()
{
	checkCudaErrors(cudaFree(gpuUFourier));
	checkCudaErrors(cudaFree(gpuUReal));
	checkCudaErrors(cudaFree(gpuUImag));

	checkCudaErrors(cudaFree(gpuFFourier));
	checkCudaErrors(cudaFree(gpuFReal));
	checkCudaErrors(cudaFree(gpuFImag));
	checkCudaErrors(cudaFree(gpuFZeroComponent));
	
	checkCudaErrors(cudaFree(gpuA));
	checkCudaErrors(cudaFree(gpuB));
	checkCudaErrors(cudaFree(gpuC));

	delete this->velPhi;
	delete this->velTheta;
	delete this->pressure;
	delete this->density;

	checkCudaErrors(cudaDeviceReset());

	//delete this->particles;
}

void KaminoSolver::setTextureParams(table2D* tex)
{
	tex->addressMode[0] = cudaAddressModeWrap;
	tex->addressMode[1] = cudaAddressModeMirror;
	tex->filterMode = cudaFilterModeLinear;
	tex->normalized = true;
}

void KaminoSolver::copyVelocity2GPU()
{
	velPhi->copyToGPU();
	velTheta->copyToGPU();
}

void KaminoSolver::copyDensity2GPU()
{
	density->copyToGPU();
}

__global__ void precomputeABCKernel
(fReal* A, fReal* B, fReal* C)
{
	int splitVal = nThetaGlobal / blockDim.x;
	int nIndex = blockIdx.x / splitVal;
	int threadSequence = blockIdx.x % splitVal;

	int i = threadIdx.x + threadSequence * blockDim.x;
	int n = nIndex - nPhiGlobal / 2;

	int index = nIndex * nThetaGlobal + i;
	fReal thetaI = (i + centeredThetaOffset) * gridLenGlobal;

	fReal cosThetaI = cosf(thetaI);
	fReal sinThetaI = sinf(thetaI);

	fReal valB = -2.0 / (gridLenGlobal * gridLenGlobal)
		- n * n / (sinThetaI * sinThetaI);
	fReal valA = 1.0 / (gridLenGlobal * gridLenGlobal)
		- cosThetaI / 2.0 / gridLenGlobal / sinThetaI;
	fReal valC = 1.0 / (gridLenGlobal * gridLenGlobal)
		+ cosThetaI / 2.0 / gridLenGlobal / sinThetaI;
	if (n != 0)
	{
		if (i == 0)
		{
			fReal coef = powf(-1.0, n);
			valB += valA;
			valA = 0.0;
		}
		if (i == nThetaGlobal - 1)
		{
			fReal coef = powf(-1.0, n);
			valB += valC;
			valC = 0.0;
		}
	}
	else
	{
		valA = 0.0;
		valB = 1.0;
		valC = 0.0;
	}
	A[index] = valA;
	B[index] = valB;
	C[index] = valC;
}

void KaminoSolver::determineLayout(dim3& gridLayout, dim3& blockLayout,
	size_t nTheta_row, size_t nPhi_col)
{
	if (nPhi_col <= this->nThreadxMax)
	{
		gridLayout = dim3(nTheta_row);
		blockLayout = dim3(nPhi_col);
	}
	else
	{
		int splitVal = (nPhi_col + nThreadxMax - 1) / nThreadxMax;

		gridLayout = dim3(nTheta_row * splitVal);
		blockLayout = dim3(nThreadxMax);
	}
}

void KaminoSolver::precomputeABCCoef()
{
	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));

	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, nPhi, nTheta);
	precomputeABCKernel<<<gridLayout, blockLayout>>>
	(this->gpuA, this->gpuB, this->gpuC);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void KaminoSolver::stepForward(fReal timeStep)
{
	this->timeStep = timeStep;

	advection();
	geometric();
	projection();

	this->timeElapsed += timeStep;
}

void KaminoSolver::swapVelocityBuffers()
{
	this->velPhi->swapGPUBuffer();
	this->velTheta->swapGPUBuffer();
}

void KaminoSolver::copyVelocityBack2CPU()
{
	this->velPhi->copyBackToCPU();
	this->velTheta->copyBackToCPU();
}

void KaminoSolver::copyDensityBack2CPU()
{
	this->density->copyBackToCPU();
}

// <<<<<<<<<<
// OUTPUT >>>>>>>>>>

void KaminoSolver::initDensityfromPic(std::string path)
{
	if (path == "")
	{
		return;
	}
	cv::Mat image_In;
	image_In = cv::imread(path, cv::IMREAD_COLOR);
	if (!image_In.data)
	{
		std::cerr << "No density image provided." << std::endl;
		return;
	}

	cv::Mat image_Flipped;
	cv::flip(image_In, image_Flipped, 1);

	cv::Mat image_Resized;
	cv::Size size(nPhi, nTheta);
	cv::resize(image_Flipped, image_Resized, size);

	for (size_t i = 0; i < nPhi; ++i)
	{
		for (size_t j = 0; j < nTheta; ++j)
		{
			cv::Point3_<uchar>* p = image_Resized.ptr<cv::Point3_<uchar>>(j, i);
			fReal B = p->x; // B
			fReal G = p->y; // G
			fReal R = p->z; // R
			this->density->setCPUValueAt(i, j, (B + G + R) / 3.0);
		}
	}

	this->density->copyToGPU();
}

void KaminoSolver::initParticlesfromPic(std::string path, size_t parPerGrid)
{
	//this->particles = new KaminoParticles(path, parPerGrid, nTheta);
}

void KaminoSolver::write_data_bgeo(const std::string& s, const int frame)
{
	std::string file = s + std::to_string(frame) + ".bgeo";
	std::cout << "Writing to: " << file << std::endl;

	Partio::ParticlesDataMutable* parts = Partio::create();
	Partio::ParticleAttribute pH, vH, densityVal;
	pH = parts->addAttribute("position", Partio::VECTOR, 3);
	vH = parts->addAttribute("v", Partio::VECTOR, 3);
	densityVal = parts->addAttribute("densityValue", Partio::FLOAT, 1);

	vec3 pos;
	vec3 vel;

	size_t iWest, iEast, jNorth, jSouth;
	fReal uWest, uEast, vNorth, vSouth;

	velPhi->copyBackToCPU();
	velTheta->copyBackToCPU();
	density->copyBackToCPU();

	for (size_t j = 0; j < nTheta; ++j)
	{
		for (size_t i = 0; i < nPhi; ++i)
		{
			iWest = i;
			uWest = velPhi->getCPUValueAt(iWest, j);
			i == (nPhi - 1) ? iEast = 0 : iEast = i + 1;
			uEast = velPhi->getCPUValueAt(iEast, j);

			if (j == 0)
			{
				jNorth = jSouth = 0;
			}
			else if (j == nTheta - 1)
			{
				jNorth = jSouth = nTheta - 2;
			}
			else
			{
				jNorth = j - 1;
				jSouth = j;
			}
			vNorth = velTheta->getCPUValueAt(i, jNorth);
			vSouth = velTheta->getCPUValueAt(i, jSouth);

			fReal velocityPhi, velocityTheta;
			velocityPhi = (uWest + uEast) / 2.0;
			velocityTheta = (vNorth + vSouth) / 2.0;

			pos = vec3((i + centeredPhiOffset) * gridLen, (j + centeredThetaOffset) * gridLen, 0.0);
			vel = vec3(0.0, velocityTheta, velocityPhi);
			mapVToSphere(pos, vel);
			mapPToSphere(pos);

			float densityValuefloat = density->getCPUValueAt(i, j);

			int idx = parts->addParticle();
			float* p = parts->dataWrite<float>(pH, idx);
			float* v = parts->dataWrite<float>(vH, idx);
			float* d = parts->dataWrite<float>(densityVal, idx);
			
			for (int k = 0; k < 3; ++k) 
			{
				p[k] = pos[k];
				v[k] = vel[k];
			}
			d[0] = densityValuefloat;
		}
	}

	Partio::write(file.c_str(), *parts);
	parts->release();
}

/*void KaminoSolver::write_particles_bgeo(const std::string& s, const int frame)
{
	std::string file = s + std::to_string(frame) + ".bgeo";
	std::cout << "Writing to: " << file << std::endl;

	Partio::ParticlesDataMutable* parts = Partio::create();
	Partio::ParticleAttribute pH, colorVal;
	pH = parts->addAttribute("position", Partio::VECTOR, 3);
	colorVal = parts->addAttribute("color", Partio::VECTOR, 3);

	vec3 pos;
	vec3 col;

	this->particles->copyBack2CPU();

	for (size_t i = 0; i < particles->numOfParticles; ++i)
	{
		pos = vec3(particles->coordCPUBuffer[2 * i],
			particles->coordCPUBuffer[2 * i + 1], 0.0);
		mapPToSphere(pos);

		col = vec3(particles->colorBGR[3 * i + 1],
			particles->colorBGR[3 * i + 2],
			particles->colorBGR[3 * i + 3]);

		int idx = parts->addParticle();
		float* p = parts->dataWrite<float>(pH, idx);
		float* c = parts->dataWrite<float>(colorVal, idx);
	
		for (int k = 0; k < 3; ++k)
		{
			p[k] = pos[k];
			c[k] = col[k];
		}
	}

	Partio::write(file.c_str(), *parts);
	parts->release();
}*/

void KaminoSolver::mapPToSphere(vec3& pos) const
{
	float theta = pos[1];
	float phi = pos[0];
	pos[0] = radius * sin(theta) * cos(phi);
	pos[2] = radius * sin(theta) * sin(phi);
	pos[1] = radius * cos(theta);
}

void KaminoSolver::mapVToSphere(vec3& pos, vec3& vel) const
{
	float theta = pos[1];
	float phi = pos[0];

	float u_theta = vel[1];
	float u_phi = vel[2];

	vel[0] = cos(theta) * cos(phi) * u_theta - sin(phi) * u_phi;
	vel[2] = cos(theta) * sin(phi) * u_theta + cos(phi) * u_phi;
	vel[1] = -sin(theta) * u_theta;
}
