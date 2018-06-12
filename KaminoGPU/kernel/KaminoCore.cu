# include "KaminoSolver.cuh"
# include "../include/KaminoGPU.cuh"
# include "../include/KaminoTimer.cuh"

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

# ifdef RUNGE_KUTTA
	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
	fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
# endif

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

# ifdef RUNGE_KUTTA
	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
	fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
# endif

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

# ifdef RUNGE_KUTTA
	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal muPhi = sampleVPhi(velPhi, midPhi, midTheta, nPitchInElements);
	fReal muTheta = sampleVTheta(velTheta, midPhi, midTheta, nPitchInElements);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
# endif

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
	fReal updatedPhi = posPhi;
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



# ifdef WRITE_VELOCITY_DATA
	determineLayout(gridLayout, blockLayout, density->getNTheta(), density->getNPhi());
	advectionCentered<<<gridLayout, blockLayout>>>
	(density->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
		density->getGPUThisStep(), density->getNextStepPitchInElements());
	
	density->swapGPUBuffer();
# endif
# ifdef WRITE_PARTICLES
	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
	advectionParticles<<<gridLayout, blockLayout>>>
	(particles->coordGPUNextStep, velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), particles->coordGPUThisStep,
		velPhi->getNextStepPitchInElements());
	particles->swapGPUBuffers();
# endif

	swapVelocityBuffers();
}

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

//nTheta by nPhi
__global__ void geometricFillKernel
(fReal* intermediateOutputPhi, fReal* intermediateOutputTheta, fReal* velPhiInput, fReal* velThetaInput,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int gridPhiId = threadIdx.x + threadSequence * blockDim.x;
	int gridThetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gTheta = ((fReal)gridThetaId + centeredThetaOffset) * gridLenGlobal;
	// The factor

	size_t phiLeft = gridPhiId;
	size_t phiRight = (gridPhiId + 1) % nPhiGlobal;
	fReal uPrev = 0.5 * (velPhiInput[phiLeft + nPitchInElements * gridThetaId]
		+ velPhiInput[phiRight + nPitchInElements * gridThetaId]);

	fReal vPrev;
	if (gridThetaId == 0)
	{
		size_t oppositePhiIdx = (gridPhiId + nPhiGlobal / 2) % nPhiGlobal;
		vPrev = 0.75 * velThetaInput[gridPhiId]
			+ 0.25 * velThetaInput[oppositePhiIdx];
	}
	else if (gridThetaId == nThetaGlobal - 1)
	{
		size_t oppositePhiIdx = (gridPhiId + nPhiGlobal / 2) % nPhiGlobal;
		vPrev = 0.75 * velThetaInput[gridPhiId + nPitchInElements * (gridThetaId - 1)]
			+ 0.25 * velThetaInput[oppositePhiIdx + nPitchInElements * (gridThetaId - 1)];
	}
	else
	{
		vPrev = 0.5 * (velThetaInput[gridPhiId + nPitchInElements * (gridThetaId - 1)]
			+ velThetaInput[gridPhiId + nPitchInElements * gridThetaId]);
	}

	fReal G = timeStepGlobal * cosf(gTheta) / (radiusGlobal * sinf(gTheta));
	fReal uNext;
	if (abs(G) > eps)
	{
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

	intermediateOutputPhi[gridPhiId + nPitchInElements * gridThetaId] = uNext;
	intermediateOutputTheta[gridPhiId + nPitchInElements * gridThetaId] = vNext;
}

//nTheta by nPhi
__global__ void assignPhiKernel(fReal* velPhiOutput, fReal* intermediateInputPhi,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	size_t phigridLeft;
	if (phiId == 0)
		phigridLeft = nPhiGlobal - 1;
	else
		phigridLeft = phiId - 1;
	velPhiOutput[phiId + nPitchInElements * thetaId] =
		0.5 * (intermediateInputPhi[phigridLeft + nPitchInElements * thetaId]
			+ intermediateInputPhi[phiId + nPitchInElements * thetaId]);
}

//nTheta - 1 by nPhi
__global__ void assignThetaKernel(fReal* velThetaOutput, fReal*intermediateInputTheta,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	velThetaOutput[phiId + nPitchInElements * thetaId] =
		0.5 * (intermediateInputTheta[phiId + nPitchInElements * thetaId]
			+ intermediateInputTheta[phiId + nPitchInElements * (thetaId + 1)]);
}

void KaminoSolver::geometric()
{
	dim3 gridLayout;
	dim3 blockLayout;
	//intermediate: pressure.this as phi, next as theta

	determineLayout(gridLayout, blockLayout, nTheta, nPhi);
	geometricFillKernel<<<gridLayout, blockLayout>>>
	(pressure->getGPUThisStep(), pressure->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
		pressure->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	determineLayout(gridLayout, blockLayout, velPhi->getNTheta(), velPhi->getNPhi());
	assignPhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), pressure->getGPUThisStep(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
	assignThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), pressure->getGPUNextStep(), velTheta->getNextStepPitchInElements());
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
# ifdef WRITE_PARTICLES
	solver.initParticlesfromPic(colorImage, this->particleDensity);
# endif

	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, &(this->radius), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->dt), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));

# ifdef WRITE_VELOCITY_DATA
	solver.write_data_bgeo(gridPath, 0);
# endif
# ifdef WRITE_PARTICLES
	solver.write_particles_bgeo(particlePath, 0);
# endif

# ifdef PERFORMANCE_BENCHMARK
	KaminoTimer timer;
	timer.startTimer();
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
# ifdef WRITE_VELOCITY_DATA
		solver.write_data_bgeo(gridPath, i);
# endif
# ifdef WRITE_PARTICLES
		solver.write_particles_bgeo(particlePath, i);
# endif
	}

# ifdef PERFORMANCE_BENCHMARK
	float gpu_time = timer.stopTimer();
# endif

	std::cout << "Time spent: " << gpu_time << "ms" << std::endl;
	std::cout << "Performance: " << 1000.0 * frames / gpu_time << " frames per second" << std::endl;
}