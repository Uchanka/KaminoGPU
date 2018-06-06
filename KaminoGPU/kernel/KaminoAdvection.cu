# include "KaminoSolver.cuh"

static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal gridLenGlobal;

__device__ fReal validateCoord(fReal& phi, fReal& theta)
{
	fReal ret = 1.0f;
	theta = theta - static_cast<int>(floorf(theta / M_2PI)) * M_2PI;
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
	phi = phi - static_cast<int>(floorf(phi / M_2PI)) * M_2PI;
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

__global__ void geometricPhi(fReal* velPhiOutput, fReal* velPhiInput, fReal* velThetaInput,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
	fReal uPhi = sampleVPhi(velPhiInput, gPhi, gTheta, nPitchInElements);
	fReal uTheta = sampleVTheta(velThetaInput, gPhi, gTheta, nPitchInElements);

	fReal uPhiPrev = velPhiInput[phiId + thetaId * nPitchInElements];
	fReal deltauPhi = -timeStepGlobal * uTheta * uPhi * cosf(gTheta) / (radiusGlobal * sinf(gTheta));
	velPhiOutput[phiId + thetaId * nPitchInElements] = deltauPhi + uPhiPrev;
}

__global__ void geometricTheta(fReal* velThetaOutput, fReal* velPhiInput, fReal* velThetaInput,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;
	fReal uPhi = sampleVPhi(velPhiInput, gPhi, gTheta, nPitchInElements);
	fReal uTheta = sampleVTheta(velThetaInput, gPhi, gTheta, nPitchInElements);

	fReal uThetaPrev = velThetaInput[phiId + thetaId * nPitchInElements];
	fReal deltauTheta = timeStepGlobal * uPhi * uPhi * cosf(gTheta) / (radiusGlobal * sinf(gTheta));
	velThetaOutput[phiId + thetaId * nPitchInElements] = deltauTheta + uThetaPrev;
}

void KaminoSolver::geometric()
{
	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
	geometricPhi<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
		velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
	geometricTheta<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
		velTheta->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	swapVelocityBuffers();
}

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

void KaminoSolver::precomputeABCCoef()
{
	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, nPhi, nTheta);
	precomputeABCKernel << <gridLayout, blockLayout >> >
		(this->gpuA, this->gpuB, this->gpuC);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

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



	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, &(this->radius), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->timeStep), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));



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
	const int fftRank = 1;
	checkCudaErrors((cudaError_t)cufftPlanMany(&kaminoPlan, fftRank, sigLenArr,
		NULL, 1, nPhi,
		NULL, 1, nPhi,
		CUFFT_C2C, nTheta));
}