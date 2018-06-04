# include "KaminoSolver.cuh"

static table2D texAdvVelPhi;
static table2D texAdvVelTheta;
static table2D texAdvDensity;

static __constant__ size_t nPhiGlobalAdv;
static __constant__ size_t nThetaGlobalAdv;
static __constant__ fReal radiusGlobalAdv;
static __constant__ fReal timeStepGlobalAdv;
static __constant__ fReal gridLenGlobalAdv;

__device__ fReal validateCoord(fReal& phi, fReal& theta)
{
	fReal ret = 1.0f;
	theta = theta - static_cast<int>(floorf(theta / M_2PI));
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
	phi = phi - static_cast<int>(floorf(phi / M_2PI));
	return ret;
}

__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha)
{
	return (1.0 - alpha) * from + alpha * to;
}

__device__ fReal sampleVPhi(fReal* input, fReal phiRaw, fReal thetaRaw, size_t pitch)
{
	fReal phi = phiRaw - gridLenGlobalAdv * vPhiPhiOffset;
	fReal theta = thetaRaw - gridLenGlobalAdv * vPhiThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLenGlobalAdv;
	fReal isFlippedPole = validateCoord(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f)
		|| thetaIndex == nThetaGlobalAdv - 1)
	{
		size_t phiLower = (phiIndex) % nPhiGlobalAdv;
		size_t phiHigher = (phiLower + 1) % nPhiGlobalAdv;
		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		phiLower = (phiIndex + nPhiGlobalAdv / 2) % nPhiGlobalAdv;
		phiHigher = (phiLower + 1) % nPhiGlobalAdv;

		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % nPhiGlobalAdv;
		size_t phiHigher = (phiLower + 1) % nPhiGlobalAdv;
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
	fReal phi = phiRaw - gridLenGlobalAdv * vThetaPhiOffset;
	fReal theta = thetaRaw - gridLenGlobalAdv * vThetaThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLenGlobalAdv;
	bool isFlippedPole = validateCoord(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f) ||
		 thetaIndex == nThetaGlobalAdv - 2)
	{
		size_t phiLower = phiIndex % nPhiGlobalAdv;
		size_t phiHigher = (phiLower + 1) % nPhiGlobalAdv;
		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		phiLower = (phiLower + nPhiGlobalAdv / 2) % nPhiGlobalAdv;
		phiHigher = (phiHigher + nPhiGlobalAdv / 2) % nPhiGlobalAdv;
		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		alphaTheta = 0.5 * alphaTheta;
		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % nPhiGlobalAdv;
		size_t phiHigher = (phiLower + 1) % nPhiGlobalAdv;
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
	fReal phi = phiRaw - gridLenGlobalAdv * centeredPhiOffset;
	fReal theta = thetaRaw - gridLenGlobalAdv * centeredThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLenGlobalAdv;
	bool isFlippedPole = validateCoord(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f) ||
		thetaIndex == nThetaGlobalAdv - 1)
	{
		size_t phiLower = phiIndex % nPhiGlobalAdv;
		size_t phiHigher = (phiLower + 1) % nPhiGlobalAdv;
		fReal lowerBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		phiLower = (phiLower + nPhiGlobalAdv / 2) % nPhiGlobalAdv;
		phiHigher = (phiHigher + nPhiGlobalAdv / 2) % nPhiGlobalAdv;
		fReal higherBelt = kaminoLerp(input[phiLower + pitch * thetaIndex],
			input[phiHigher + pitch * thetaIndex], alphaPhi);

		alphaTheta = 0.5 * alphaTheta;
		fReal lerped = kaminoLerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % nPhiGlobalAdv;
		size_t phiHigher = (phiLower + 1) % nPhiGlobalAdv;
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
	int splitVal = nPhiGlobalAdv / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobalAdv;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobalAdv;
	
	// Sample the speed
	fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
	fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);

	fReal latRadius = radiusGlobalAdv * sinf(gTheta);
	fReal cofPhi = timeStepGlobalAdv / latRadius;
	fReal cofTheta = timeStepGlobalAdv / radiusGlobalAdv;

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
	int splitVal = nPhiGlobalAdv / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobalAdv;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobalAdv;

	// Sample the speed
	fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
	fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);

	fReal latRadius = radiusGlobalAdv * sinf(gTheta);
	fReal cofPhi = timeStepGlobalAdv / latRadius;
	fReal cofTheta = timeStepGlobalAdv / radiusGlobalAdv;

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
	int splitVal = nPhiGlobalAdv / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + centeredPhiOffset) * gridLenGlobalAdv;
	fReal gTheta = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobalAdv;

	// Sample the speed
	fReal guPhi = sampleVPhi(velPhi, gPhi, gTheta, nPitchInElements);
	fReal guTheta = sampleVTheta(velTheta, gPhi, gTheta, nPitchInElements);

	fReal latRadius = radiusGlobalAdv * sinf(gTheta);
	fReal cofPhi = timeStepGlobalAdv / latRadius;
	fReal cofTheta = timeStepGlobalAdv / radiusGlobalAdv;

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

	fReal latRadius = radiusGlobalAdv * sinf(posTheta);
	fReal cofPhi = timeStepGlobalAdv / latRadius;
	fReal cofTheta = timeStepGlobalAdv / radiusGlobalAdv;

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
	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobalAdv, &(this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobalAdv, &(this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(radiusGlobalAdv, &(this->radius), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobalAdv, &(this->timeStep), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobalAdv, &(this->gridLen), sizeof(fReal)));
	


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