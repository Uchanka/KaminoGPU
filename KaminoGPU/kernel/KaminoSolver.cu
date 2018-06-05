# include "../include/KaminoSolver.cuh"
# include "../opencv_headers/opencv2/opencv.hpp"

// CONSTRUCTOR / DESTRUCTOR >>>>>>>>>>

const int fftRank = 1;
static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal gridLenGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal radiusGlobal;

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

	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, &(this->radius), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->timeStep), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));
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

///Advection

__device__ fReal validateCoord(fReal& phi, fReal& theta)
{
	fReal ret = 1.0f;
	theta = theta - floorf(theta / M_2PI);
	if (theta > M_PI)
	{
		theta = M_2PI - theta;
		phi += M_PI;
		ret = -ret;
	}
	if (theta < 0)
	{
		theta = -theta;
		phi += M_PI;
		ret = -ret;
	}
	phi = phi - floorf(phi / M_2PI);
	return ret;
}

__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha)
{
	return (1.0 - alpha) * from + alpha * to;
}

__device__ fReal sampleVPhi(fReal* input, fReal phiRaw, fReal thetaRaw, size_t pitch)
{
	fReal phi = phiRaw - gridLenGlobal * vPhiPhiOffset;
	fReal theta = thetaRaw - gridLenGlobal * vPhiThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLenGlobal;
	fReal isFlippedPole = validateCoord(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f)
		|| thetaIndex == nThetaGlobal - 1)
	{
		size_t phiLower = (phiIndex) % nPhiGlobal;
		size_t phiHigher = (phiLower + 1) % nPhiGlobal;
		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		phiLower = (phiIndex + nPhiGlobal / 2) % nPhiGlobal;
		phiHigher = (phiLower + 1) % nPhiGlobal;

		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % nPhiGlobal;
		size_t phiHigher = (phiLower + 1) % nPhiGlobal;
		size_t thetaLower = thetaIndex;
		size_t thetaHigher = thetaIndex + 1;

		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
			input[phiHigher + pitch * thetaLower], alphaPhi);
		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
			input[phiHigher + pitch * thetaHigher], alphaPhi);

		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
}

__device__ fReal sampleVTheta(fReal* input, fReal phiRaw, fReal thetaRaw, size_t pitch)
{
	fReal phi = phiRaw - gridLenGlobal * vThetaPhiOffset;
	fReal theta = thetaRaw - gridLenGlobal * vThetaThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLenGlobal;
	bool isFlippedPole = validateCoord(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f) ||
		thetaIndex == nThetaGlobal - 2)
	{
		size_t phiLower = phiIndex % nPhiGlobal;
		size_t phiHigher = (phiLower + 1) % nPhiGlobal;
		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
		phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		alphaTheta = 0.5 * alphaTheta;
		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % nPhiGlobal;
		size_t phiHigher = (phiLower + 1) % nPhiGlobal;
		size_t thetaLower = thetaIndex;
		size_t thetaHigher = thetaIndex + 1;

		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
			input[phiHigher + pitch * thetaLower], alphaPhi);
		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
			input[phiHigher + pitch * thetaHigher], alphaPhi);

		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
}

__device__ fReal sampleCentered(fReal* input, fReal phiRaw, fReal thetaRaw, size_t pitch)
{
	fReal phi = phiRaw - gridLenGlobal * centeredPhiOffset;
	fReal theta = thetaRaw - gridLenGlobal * centeredThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLenGlobal;
	bool isFlippedPole = validateCoord(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f) ||
		thetaIndex == nThetaGlobal - 1)
	{
		size_t phiLower = phiIndex % nPhiGlobal;
		size_t phiHigher = (phiLower + 1) % nPhiGlobal;
		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		phiLower = (phiLower + nPhiGlobal / 2) % nPhiGlobal;
		phiHigher = (phiHigher + nPhiGlobal / 2) % nPhiGlobal;
		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		alphaTheta = 0.5 * alphaTheta;
		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % nPhiGlobal;
		size_t phiHigher = (phiLower + 1) % nPhiGlobal;
		size_t thetaLower = thetaIndex;
		size_t thetaHigher = thetaIndex + 1;

		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaLower],
			input[phiHigher + pitch * thetaLower], alphaPhi);
		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaHigher],
			input[phiHigher + pitch * thetaHigher], alphaPhi);

		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
}

__global__ void advectionVPhiKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;

	// Sample the speed
	fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
	fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);

	fReal latRadius = radiusGlobal * sinf(gTheta);
	fReal cofPhi = timeStepGlobal / latRadius;
	fReal cofTheta = timeStepGlobal / radiusGlobal;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
	fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;

	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;

	fReal advectedVal = sampleVPhi(velPhi, pPhi, pTheta, nPitchInElements);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
};

__global__ void advectionVThetaKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;

	// Sample the speed
	fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
	fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);

	fReal latRadius = radiusGlobal * sinf(gTheta);
	fReal cofPhi = timeStepGlobal / latRadius;
	fReal cofTheta = timeStepGlobal / radiusGlobal;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
	fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;

	fReal advectedVal = sampleVTheta(velTheta, pPhi, pTheta, nPitchInElements);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
}

__global__ void advectionCentered
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta, fReal* attributeInput, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + centeredPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;

	// Sample the speed
	fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
	fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);

	fReal latRadius = radiusGlobal * sinf(gTheta);
	fReal cofPhi = timeStepGlobal / latRadius;
	fReal cofTheta = timeStepGlobal / radiusGlobal;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
	fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;

	fReal advectedVal = sampleCentered(attributeInput, pPhi, pTheta, nPitchInElements);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
};

__global__ void advectionParticles(fReal* output, fReal* velPhi, fReal* velTheta, fReal* input, size_t nPitch)
{
	int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	fReal posPhi = input[2 * particleId];
	fReal posTheta = input[2 * particleId + 1];

	fReal uPhi = sampleVPhi(velPhi, posPhi, posTheta, nPitch);
	fReal uTheta = sampleVTheta(velTheta, posPhi, posTheta, nPitch);

	fReal latRadius = radiusGlobal * sinf(posTheta);
	fReal cofPhi = timeStepGlobal / latRadius;
	fReal cofTheta = timeStepGlobal / radiusGlobal;

	fReal updatedTheta = posTheta + uTheta * cofTheta;
	fReal updatedPhi = 0.0;
	if (latRadius > 1e-7f)
		updatedPhi = posPhi + uPhi * cofPhi;
	validateCoord(updatedPhi, updatedTheta);

	output[2 * particleId] = updatedPhi;
	output[2 * particleId + 1] = updatedTheta;
}

void KaminoSolver::advection()
{
	///kernel call goes here
	// Advect Phi
	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
	advectionVPhiKernel<<<gridLayout, blockLayout>>>
		(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	// Advect Theta
	determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
	advectionVThetaKernel<<<gridLayout, blockLayout>>>
		(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



# ifdef WRITE_BGEO
	determineLayout(gridLayout, blockLayout, density->getNTheta(), density->getNPhi());
	advectionCentered<<<gridLayout, blockLayout>>>
		(density->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
			density->getGPUThisStep(), density->getNextStepPitchInElements());

	density->swapGPUBuffer();

	/*determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
	advectionParticles<<<gridLayout, blockLayout>>>
	(particles->coordGPUNextStep, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), particles->coordGPUThisStep,
	velPhi->getNextStepPitchInElements());
	particles->swapGPUBuffers();*/
# endif

	swapVelocityBuffers();
}

///GEOMETRIC

__device__ fReal _root3(fReal x)
{
	fReal s = 1.;
	while (x < 1.)
	{
		x *= 8.;
		s *= 0.5;
	}
	while (x > 8.)
	{
		x *= 0.125;
		s *= 2.;
	}
	fReal r = 1.5;
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	return r * s;
}

__device__ fReal root3(double x)
{
	if (x > 0)
		return _root3(x);
	else if (x < 0)
		return -_root3(-x);
	else
		return 0.0;
}

#define eps 1e-7f

__device__ fReal solveCubic(fReal a, fReal b, fReal c)
{
	fReal a2 = a * a;
	fReal q = (a2 - 3 * b) / 9.0;
	//q = q >= 0.0 ? q : -q;
	fReal r = (a * (2.0 * a2 - 9.0 * b) + 27.0 * c) / 54.0;

	fReal r2 = r * r;
	fReal q3 = q * q * q;
	fReal A, B;
	if (r2 <= (q3 + eps))
	{
		double t = r / sqrtf(q3);
		if (t < -1)
			t = -1;
		if (t > 1)
			t = 1;
		t = acosf(t);
		a /= 3.0;
		q = -2.0 * sqrtf(q);
		return q * cosf(t / 3.0) - a;
	}
	else
	{
		A = -root3(fabsf(r) + sqrtf(r2 - q3));
		if (r < 0)
			A = -A;

		B = A == 0 ? 0 : B = q / A;

		a /= 3.0;
		return (A + B) - a;
	}
}

//nTheta - 1 by nPhi
__global__ void geometricKernel
(fReal* velPhiOutput, fReal* velThetaOutput, fReal* velPhiInput, fReal* velThetaInput,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
	// The factor
	fReal factor = timeStepGlobal * cosf(gTheta) / (radiusGlobal * sinf(gTheta));

	fReal uPrev = velPhiInput[phiId + nPitchInElements * (thetaId + 1)];
	fReal vPrev = velThetaInput[phiId + nPitchInElements * thetaId];

	fReal G;
	fReal uNext;
	if (abs(sinf(gTheta)) < eps)
	{
		G = timeStepGlobal * cosf(gTheta) / (radiusGlobal * sinf(gTheta));
		fReal cof = G * G;
		fReal A = 0.0;
		fReal B = (G * vPrev + 1.0) / cof;
		fReal C = -uPrev / cof;

		uNext = solveCubic(A, B, C);
	}
	else
	{
		uNext = uPrev;
	}

	fReal vNext = vPrev + G * uNext * uNext;

	velPhiOutput[(thetaId + 1) * nPitchInElements + phiId] = uNext;
	velThetaOutput[thetaId * nPitchInElements + phiId] = vNext;
}

//1 by nPhi
__global__ void copyKernel(fReal* velPhiOutput, fReal* velPhiInput,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	velPhiOutput[phiId] = velPhiInput[phiId];
	velPhiOutput[phiId + (nThetaGlobal - 1) * nPitchInElements]
		= velPhiInput[phiId + (nThetaGlobal - 1) * nPitchInElements];
}

void KaminoSolver::geometric()
{
	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, nTheta - 1, nPhi);
	geometricKernel<<<gridLayout, blockLayout>>>
		(velPhi->getGPUNextStep(), velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
			velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	determineLayout(gridLayout, blockLayout, 1, nPhi);
	copyKernel<<<gridLayout, blockLayout>>>
		(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	swapVelocityBuffers();
}

///Projection

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x);

__global__ void fillDivergenceKernel
(ComplexFourier* outputF, fReal* velPhi, fReal* velTheta,
	size_t velPhiPitchInElements, size_t velThetaPitchInElements)
{
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int gridPhiId = threadIdx.x + threadSequence * blockDim.x;
	int gridThetaId = blockIdx.x / splitVal;
	//fReal gridPhiCoord = ((fReal)gridPhiId + centeredPhiOffset) * gridLen;
	fReal gridThetaCoord = ((fReal)gridThetaId + centeredThetaOffset) * gridLenGlobal;

	fReal uEast = 0.0;
	fReal uWest = 0.0;
	fReal vNorth = 0.0;
	fReal vSouth = 0.0;

	fReal halfStep = 0.5 * gridLenGlobal;

	fReal thetaSouth = gridThetaCoord + halfStep;
	fReal thetaNorth = gridThetaCoord - halfStep;

	int phiIdWest = gridPhiId;
	int phiIdEast = (phiIdWest + 1) % nPhiGlobal;

	uWest = velPhi[gridThetaId * velPhiPitchInElements + phiIdWest];
	uEast = velPhi[gridThetaId * velPhiPitchInElements + phiIdEast];

	if (gridThetaId != 0)
	{
		int thetaNorthIdx = gridThetaId - 1;
		vNorth = velTheta[thetaNorthIdx * velThetaPitchInElements + gridPhiId];
	}
	if (gridThetaId != nThetaGlobal - 1)
	{
		int thetaSouthIdx = gridThetaId;
		vSouth = velTheta[thetaSouthIdx * velThetaPitchInElements + gridPhiId];
	}

	fReal invGridSine = 1.0 / sinf(gridThetaCoord);
	fReal sinNorth = sinf(thetaNorth);
	fReal sinSouth = sinf(thetaSouth);
	fReal factor = invGridSine / gridLenGlobal;
	fReal termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
	fReal termPhi = factor * (uEast - uWest);

	fReal div = termTheta + termPhi;

	ComplexFourier f;
	f.x = div;
	f.y = 0.0;
	outputF[gridThetaId * nPhiGlobal + gridPhiId] = f;
}

__global__ void shiftFKernel
(ComplexFourier* FFourierInput, fReal* FFourierShiftedReal, fReal* FFourierShiftedImag)
{
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int nIdx = threadIdx.x + threadSequence * blockDim.x;
	int thetaIdx = blockIdx.x / splitVal;

	int fftIndex = nPhiGlobal / 2 - nIdx;
	if (fftIndex < 0)
		fftIndex += nPhiGlobal;
	//FFourierShifted[thetaIdx * nPhi + phiIdx] = FFourierInput[thetaIdx * nPhi + fftIndex];
	fReal real = FFourierInput[thetaIdx * nPhiGlobal + fftIndex].x / (fReal)nPhiGlobal;
	fReal imag = FFourierInput[thetaIdx * nPhiGlobal + fftIndex].y / (fReal)nPhiGlobal;
	FFourierShiftedReal[nIdx * nThetaGlobal + thetaIdx] = real;
	FFourierShiftedImag[nIdx * nThetaGlobal + thetaIdx] = imag;
}

__global__ void copy2UFourier
(ComplexFourier* UFourierOutput, fReal* UFourierReal, fReal* UFourierImag)
{
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int nIdx = threadIdx.x + threadSequence * blockDim.x;
	int thetaIdx = blockIdx.x / splitVal;

	ComplexFourier u;
	u.x = UFourierReal[nIdx * nThetaGlobal + thetaIdx];
	u.y = UFourierImag[nIdx * nThetaGlobal + thetaIdx];
	UFourierOutput[thetaIdx * nPhiGlobal + nIdx] = u;
}

__global__ void cacheZeroComponents
(fReal* zeroComponentCache, ComplexFourier* input)
{
	int splitVal = nThetaGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int thetaIdx = threadIdx.x + threadSequence * blockDim.x;

	zeroComponentCache[thetaIdx] = input[thetaIdx * nPhiGlobal + nPhiGlobal / 2].x;
}

__global__ void shiftUKernel
(ComplexFourier* UFourierInput, fReal* pressure, fReal* zeroComponentCache,
	size_t nPressurePitchInElements)
{
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiIdx = threadIdx.x + threadSequence * blockDim.x;
	int thetaIdx = blockIdx.x / splitVal;

	int fftIndex = 0;
	fReal zeroComponent = zeroComponentCache[thetaIdx];
	if (phiIdx != 0)
		fftIndex = nPhiGlobal - phiIdx;
	fReal pressureVal;

	if (phiIdx % 2 == 0)
		pressureVal = UFourierInput[thetaIdx * nPhiGlobal + fftIndex].x - zeroComponent;
	else
		pressureVal = -UFourierInput[thetaIdx * nPhiGlobal + fftIndex].x - zeroComponent;

	pressure[thetaIdx * nPressurePitchInElements + phiIdx] = pressureVal;
}

__global__ void applyPressureTheta
(fReal* output, fReal* prev, fReal* pressure,
	size_t nPitchInElementsPressure, size_t nPitchInElementsVTheta)
{
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	int pressureThetaNorthId = thetaId;
	int pressureThetaSouthId = thetaId + 1;
	fReal pressureNorth = pressure[pressureThetaNorthId * nPitchInElementsPressure + phiId];
	fReal pressureSouth = pressure[pressureThetaSouthId * nPitchInElementsPressure + phiId];

	fReal deltaVTheta = (pressureSouth - pressureNorth) / (-gridLenGlobal);
	fReal previousVTheta = prev[thetaId * nPitchInElementsVTheta + phiId];
	output[thetaId * nPitchInElementsVTheta + phiId] = previousVTheta + deltaVTheta;
}
__global__ void applyPressurePhi
(fReal* output, fReal* prev, fReal* pressure,
	size_t nPitchInElementsPressure, size_t nPitchInElementsVPhi)
{
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	int pressurePhiWestId;
	if (phiId == 0)
		pressurePhiWestId = nPhiGlobal - 1;
	else
		pressurePhiWestId = phiId - 1;
	int pressurePhiEastId = phiId;

	fReal pressureWest = pressure[thetaId * nPitchInElementsPressure + pressurePhiWestId];
	fReal pressureEast = pressure[thetaId * nPitchInElementsPressure + pressurePhiEastId];

	fReal thetaBelt = (thetaId + centeredThetaOffset) * gridLenGlobal;
	fReal deltaVPhi = (pressureEast - pressureWest) / (-gridLenGlobal * sinf(thetaBelt));
	fReal previousVPhi = prev[thetaId * nPitchInElementsVPhi + phiId];
	output[thetaId * nPitchInElementsVPhi + phiId] = previousVPhi + deltaVPhi;
}

void KaminoSolver::projection()
{
	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
	fillDivergenceKernel<<<gridLayout, blockLayout>>>
		(this->gpuFFourier, this->velPhi->getGPUThisStep(), this->velTheta->getGPUThisStep(),
			this->velPhi->getThisStepPitchInElements(), this->velTheta->getThisStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	// Note that cuFFT inverse returns results are SigLen times larger
	checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
		this->gpuFFourier, this->gpuFFourier, CUFFT_INVERSE));
	checkCudaErrors(cudaGetLastError());



	// Siglen is nPhi
	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
	shiftFKernel<<<gridLayout, blockLayout>>>
		(gpuFFourier, gpuFReal, gpuFImag);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// Now gpuFDivergence stores all the Fn



	gridLayout = dim3(nPhi);
	blockLayout = dim3(nTheta / 2);
	const unsigned sharedMemSize = nTheta * 5 * sizeof(fReal);
	crKernel<<<gridLayout, blockLayout, sharedMemSize>>>
		(this->gpuA, this->gpuB, this->gpuC, this->gpuFReal, this->gpuUReal);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gridLayout = dim3(nPhi);
	blockLayout = dim3(nTheta / 2);
	crKernel<<<gridLayout, blockLayout, sharedMemSize>>>
		(this->gpuA, this->gpuB, this->gpuC, this->gpuFImag, this->gpuUImag);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
	copy2UFourier<<<gridLayout, blockLayout>>>
		(this->gpuUFourier, this->gpuUReal, this->gpuUImag);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	determineLayout(gridLayout, blockLayout, 1, nTheta);
	cacheZeroComponents<<<gridLayout, blockLayout>>>
		(gpuFZeroComponent, gpuUFourier);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
		this->gpuUFourier, this->gpuUFourier, CUFFT_FORWARD));
	checkCudaErrors(cudaGetLastError());



	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
	shiftUKernel<<<gridLayout, blockLayout>>>
		(gpuUFourier, pressure->getGPUThisStep(), this->gpuFZeroComponent,
			pressure->getThisStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//pressure->copyBackToCPU();

	determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
	applyPressureTheta<<<gridLayout, blockLayout>>>
		(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), pressure->getGPUThisStep(),
			pressure->getThisStepPitchInElements(), velTheta->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
	applyPressurePhi<<<gridLayout, blockLayout>>>
		(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), pressure->getGPUThisStep(),
			pressure->getThisStepPitchInElements(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	swapVelocityBuffers();
}