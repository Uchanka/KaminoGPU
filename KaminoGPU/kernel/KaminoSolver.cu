# include "../include/KaminoSolver.h"
# include "../include/CubicSolver.h"

// CONSTRUCTOR / DESTRUCTOR >>>>>>>>>>

const int fftRank = 1;

KaminoSolver::KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal frameDuration,
	fReal A, int B, int C, int D, int E) :
	nPhi(nPhi), nTheta(nTheta), radius(radius), gridLen(M_2PI / nPhi), invGridLen(1.0 / gridLen), frameDuration(frameDuration),
	timeStep(0.0), timeElapsed(0.0),
	A(A), B(B), C(C), D(D), E(E)
{
	/// Replace it later with functions from helper_cuda.h!
	checkCudaErrors(cudaSetDevice(0));

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

	/*this->cpuGridTypesBuffer = new gridType[nPhi * nTheta];
	checkCudaErrors(cudaMalloc((void **)(this->gpuGridTypes),
		sizeof(gridType) * nPhi * nTheta));*/

	initialize_velocity();
	copyVelocity2GPU();

	initialize_boundary();
	//copyGridType2GPU();

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
	
	checkCudaErrors(cudaFree(gpuA));
	checkCudaErrors(cudaFree(gpuB));
	checkCudaErrors(cudaFree(gpuC));

	delete this->velPhi;
	delete this->velTheta;
	delete this->pressure;

	//delete[] cpuGridTypesBuffer;
	//checkCudaErrors(cudaFree(gpuGridTypes));
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

__global__ void precomputeABCKernel
(fReal* A, fReal* B, fReal* C, fReal gridLen, int nPhi, int nTheta)
{
	int nIndex = blockIdx.x;
	int n = nIndex - nPhi / 2;
	int i = threadIdx.x;
	int index = nIndex * nTheta + i;
	fReal thetaI = (i + centeredThetaOffset) * gridLen;

	fReal cosThetaI = cosf(thetaI);
	fReal sinThetaI = sinf(thetaI);

	if (n != 0)
	{
		A[index] = 1.0 / (gridLen * gridLen)
			- 0.5 * cosThetaI / gridLen / sinThetaI;
		B[index] = -2.0 / (gridLen * gridLen) - n * n / (sinThetaI * sinThetaI);
		C[index] = 1.0 / (gridLen * gridLen) + 0.5 * cosThetaI / gridLen / sinThetaI;
	}
	else
	{
		A[index] = 0.0;
		B[index] = 1.0;
		C[index] = 0.0;
	}
}

void KaminoSolver::precomputeABCCoef()
{
	dim3 gridLayout = dim3(nPhi);
	dim3 blockLayout = dim3(nTheta);
	precomputeABCKernel<<<gridLayout, blockLayout>>>
	(this->gpuA, this->gpuB, this->gpuC, gridLen, nPhi, nTheta);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void KaminoSolver::stepForward(fReal timeStep)
{
	this->timeStep = timeStep;
	advection();
	std::cout << "Advection completed" << std::endl;
	//geometric();
	//std::cout << "Geometric completed" << std::endl;
	//bodyForce();
	//std::cout << "Body force application completed" << std::endl;
	//projection();
	//std::cout << "Projection completed" << std::endl;
	this->timeElapsed += timeStep;

	velPhi->copyBackToCPU();
	velTheta->copyBackToCPU();
}

// Phi: 0 - 2pi  Theta: 0 - pi
bool validatePhiTheta(fReal & phi, fReal & theta)
{
	int loops = static_cast<int>(std::floor(theta / M_2PI));
	theta = theta - loops * M_2PI;
	// Now theta is in 0-2pi range

	bool isFlipped = false;

	if (theta > M_PI)
	{
		theta = M_2PI - theta;
		phi += M_PI;
		isFlipped = true;
	}

	loops = static_cast<int>(std::floor(phi / M_2PI));
	phi = phi - loops * M_2PI;
	// Now phi is in 0-2pi range

	return isFlipped;
}

void KaminoSolver::bodyForce()
{
	/// This is just a place holder now...
}

/* Tri-diagonal matrix solver */
void KaminoSolver::TDMSolve(fReal* a, fReal* b, fReal* c, fReal* d)
{
	// |b0 c0 0 ||x0| |d0|
	// |a1 b1 c1||x1|=|d1|
	// |0  a2 b2||x2| |d2|

	int n = nTheta;
	n--; // since we index from 0
	c[0] /= b[0];
	d[0] /= b[0];

	for (int i = 1; i < n; i++) {
		c[i] /= b[i] - a[i] * c[i - 1];
		d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1]);
	}

	d[n] = (d[n] - a[n] * d[n - 1]) / (b[n] - a[n] * c[n - 1]);

	for (int i = n; i-- > 0;) {
		d[i] -= c[i] * d[i + 1];
	}
}

gridType KaminoSolver::getGridTypeAt(size_t x, size_t y)
{
	return this->cpuGridTypesBuffer[getIndex(x, y)];
}

/*KaminoQuantity* KaminoSolver::getAttributeNamed(std::string name)
{
	return (*this)[name];
}*/

void KaminoSolver::swapAttrBuffers()
{
	this->velPhi->swapGPUBuffer();
	this->velTheta->swapGPUBuffer();
}

void KaminoSolver::copyVelocityBack2CPU()
{
	this->velPhi->copyBackToCPU();
	this->velTheta->copyBackToCPU();
}


// <<<<<<<<<<
// OUTPUT >>>>>>>>>>


void KaminoSolver::write_data_bgeo(const std::string& s, const int frame)
{
	std::string file = s + std::to_string(frame) + ".bgeo";
	std::cout << "Writing to: " << file << std::endl;

	Partio::ParticlesDataMutable* parts = Partio::create();
	Partio::ParticleAttribute pH, vH;// , psH, dens;
	pH = parts->addAttribute("position", Partio::VECTOR, 3);
	vH = parts->addAttribute("v", Partio::VECTOR, 3);

	vec3 pos;
	vec3 vel;

	size_t iWest, iEast, jNorth, jSouth;
	fReal uWest, uEast, vNorth, vSouth;

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

			int idx = parts->addParticle();
			float* p = parts->dataWrite<float>(pH, idx);
			float* v = parts->dataWrite<float>(vH, idx);
			
			for (int k = 0; k < 3; ++k) 
			{
				p[k] = pos[k];
				v[k] = vel[k];
			}
		}
	}

	Partio::write(file.c_str(), *parts);
	parts->release();
}

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
