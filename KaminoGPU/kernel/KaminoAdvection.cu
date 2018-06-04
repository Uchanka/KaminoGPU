# include "KaminoSolver.cuh"

static table2D texAdvVelPhi;
static table2D texAdvVelTheta;
static table2D texAdvDensity;

static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal gridLenGlobal;

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

__global__ void advectionParticles(fReal* output, fReal* velPhi, fReal* velTheta, fReal* input)
{
	int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	fReal posPhi = input[2 * particleId];
	fReal posTheta = input[2 * particleId + 1];

	fReal uPhi = sampleVPhi(velPhi, posPhi, posTheta);
	fReal uTheta = sampleVTheta(velTheta, posPhi, posTheta);

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
	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, &(this->radius), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->timeStep), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));
	


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
	advectionCentered//<<<gridLayout, blockLayout>>>
	(density->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
		density->getGPUThisStep(), density->getNextStepPitchInElements());
	
	density->swapGPUBuffer();

	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
	advectionParticles<<<gridLayout, blockLayout>>>
	(particles->coordGPUNextStep, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), particles->coordGPUThisStep);
	particles->swapGPUBuffers();
# endif

	swapVelocityBuffers();
}