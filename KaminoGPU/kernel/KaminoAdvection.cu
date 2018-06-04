# include "KaminoSolver.cuh"

static table2D texAdvVelPhi;
static table2D texAdvVelTheta;
static table2D texAdvDensity;

static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal gridLenGlobal;

__device__ fReal validateTex(fReal& phiTex, fReal& thetaTex)
{
	fReal returnBit = 1.0;
	
	if (thetaTex >= 1.0)
	{
		fReal pred = thetaTex - 1.0;
		thetaTex = 2.0 - thetaTex;
		if (pred >= gridLenGlobal / vThetaThetaNorm)
		{
			phiTex += 0.5;
			returnBit = -returnBit;
		}
	}
	if (thetaTex < 0.0)
	{
		thetaTex = -thetaTex;
		if (thetaTex >= gridLenGlobal / vThetaThetaNorm)
		{
			phiTex += 0.5;
			returnBit = -returnBit;
		}
	}

	phiTex -= floorf(phiTex);

	return returnBit;
}

__global__ void advectionVPhiKernel
	(fReal* attributeOutput, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
    int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
	// Coord in u-v texture space
	fReal gPhiTex = (gPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal gThetaTex = (gTheta - vPhiThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	// Sample the speed
	fReal guPhi = tex2D<fReal>(texAdvVelPhi, gPhiTex, gThetaTex);
	gPhiTex = (gPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	gThetaTex = (gTheta - vThetaThetaOffset * gridLenGlobal) / vThetaThetaNorm;
	fReal guTheta = tex2D<fReal>(texAdvVelTheta, gPhiTex, gThetaTex);

	fReal latRadius = radiusGlobal * sinf(gTheta);
	fReal cofPhi = timeStepGlobal / latRadius;
	fReal cofTheta = timeStepGlobal / radiusGlobal;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal midPhiTex = (midPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal midThetaTex = (midTheta - vPhiThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	fReal ret = validateTex(midPhiTex, midThetaTex);
	fReal muPhi = ret * tex2D<fReal>(texAdvVelPhi, midPhiTex, midThetaTex);
	midPhiTex = (midPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	midThetaTex = (midTheta - vThetaThetaOffset * gridLenGlobal) / vThetaThetaNorm;
	ret = validateTex(midPhiTex, midThetaTex);
	fReal muTheta = ret * tex2D<fReal>(texAdvVelTheta, midPhiTex, midThetaTex);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;
	fReal pPhiTex = (pPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal pThetaTex = (pTheta - vPhiThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	ret = validateTex(pPhiTex, pThetaTex);

	fReal advectedVal = ret * tex2D<fReal>(texAdvVelPhi, pPhiTex, pThetaTex);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
};

__global__ void advectionVThetaKernel
(fReal* attributeOutput, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;

	// Coord in u-v texture space
	fReal gPhiTex = (gPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal gThetaTex = (gTheta - vPhiThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	// Sample the speed
	fReal guPhi = tex2D<fReal>(texAdvVelPhi, gPhiTex, gThetaTex);
	gPhiTex = (gPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	gThetaTex = (gTheta - vThetaThetaOffset * gridLenGlobal) / vThetaThetaNorm;
	fReal guTheta = tex2D<fReal>(texAdvVelTheta, gPhiTex, gThetaTex);

	fReal latRadius = radiusGlobal * sinf(gTheta);
	fReal cofPhi = timeStepGlobal / latRadius;
	fReal cofTheta = timeStepGlobal / radiusGlobal;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal midPhiTex = (midPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal midThetaTex = (midTheta - vPhiThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	fReal ret = validateTex(midPhiTex, midThetaTex);
	fReal muPhi = ret * tex2D(texAdvVelPhi, midPhiTex, midThetaTex);
	midPhiTex = (midPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	midThetaTex = (midTheta - vThetaThetaOffset * gridLenGlobal) / vThetaThetaNorm;
	ret = validateTex(midPhiTex, midThetaTex);
	fReal muTheta = ret * tex2D(texAdvVelTheta, midPhiTex, midThetaTex);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;
	fReal pPhiTex = (pPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	fReal pThetaTex = (pTheta - vThetaThetaOffset * gridLenGlobal) / vThetaThetaNorm;
	ret = validateTex(pPhiTex, pThetaTex);

	fReal advectedVal = ret * tex2D<fReal>(texAdvVelTheta, pPhiTex, pThetaTex);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
}

__global__ void advectionCentered
(fReal* attributeOutput, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + centeredPhiOffset) * gridLenGlobal;
	fReal gTheta = ((fReal)thetaId + centeredThetaOffset) * gridLenGlobal;

	// Coord in u-v texture space
	fReal gPhiTex = (gPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal gThetaTex = (gTheta - vPhiThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	// Sample the speed
	fReal guPhi = tex2D<fReal>(texAdvVelPhi, gPhiTex, gThetaTex);
	gPhiTex = (gPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	gThetaTex = (gTheta - vThetaThetaOffset * gridLenGlobal) / vThetaThetaNorm;
	fReal guTheta = tex2D<fReal>(texAdvVelTheta, gPhiTex, gThetaTex);

	fReal latRadius = radiusGlobal * sinf(gTheta);
	fReal cofPhi = timeStepGlobal / latRadius;
	fReal cofTheta = timeStepGlobal / radiusGlobal;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal midPhiTex = (midPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal midThetaTex = (midTheta - vPhiThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	fReal ret = validateTex(midPhiTex, midThetaTex);
	fReal muPhi = ret * tex2D<fReal>(texAdvVelPhi, midPhiTex, midThetaTex);
	midPhiTex = (midPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	midThetaTex = (midTheta - vThetaThetaOffset * gridLenGlobal) / vThetaThetaNorm;
	ret = validateTex(midPhiTex, midThetaTex);
	fReal muTheta = ret * tex2D<fReal>(texAdvVelTheta, midPhiTex, midThetaTex);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;
	fReal pPhiTex = (pPhi - centeredPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal pThetaTex = (pTheta - centeredThetaOffset * gridLenGlobal) / vPhiThetaNorm;
	ret = validateTex(pPhiTex, pThetaTex);

	fReal advectedVal = tex2D<fReal>(texAdvDensity, pPhiTex, pThetaTex);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
};

__device__ void validateCoord(fReal& phi, fReal& theta)
{
	theta = theta - floorf(theta / M_2PI);
	if (theta > M_PI)
	{
		theta = M_2PI - theta;
		phi += M_PI;
	}
	if (theta < 0)
	{
		theta = -theta;
		phi += M_PI;
	}
	phi = phi - floorf(phi / M_2PI);
}

__global__ void advectionParticles(fReal* output, fReal* input)
{
	int particleId = blockIdx.x * blockDim.x + threadIdx.x;
	fReal posPhi = input[2 * particleId];
	fReal posTheta = input[2 * particleId + 1];

	fReal phiTex = (posPhi - vPhiPhiOffset * gridLenGlobal) / vPhiPhiNorm;
	fReal pThetaTex = (posTheta - vPhiPhiOffset * gridLenGlobal) / vPhiThetaNorm;
	fReal uPhi = tex2D<fReal>(texAdvVelPhi, phiTex, pThetaTex);
	phiTex = (posPhi - vThetaPhiOffset * gridLenGlobal) / vThetaPhiNorm;
	pThetaTex = (posTheta - vThetaPhiOffset * gridLenGlobal) / vThetaThetaNorm;
	fReal uTheta = tex2D<fReal>(texAdvVelTheta, phiTex, pThetaTex);

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
	setTextureParams(&texAdvVelPhi);
	setTextureParams(&texAdvVelTheta);
	velPhi->bindTexture(&texAdvVelPhi);
	velTheta->bindTexture(&texAdvVelTheta);



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
	(velPhi->getGPUNextStep(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	// Advect Theta
	determineLayout(gridLayout, blockLayout, velTheta->getNTheta(), velTheta->getNPhi());
	advectionVThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velTheta->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



# ifdef WRITE_BGEO
	setTextureParams(&texAdvDensity);
	density->bindTexture(&texAdvDensity);
	determineLayout(gridLayout, blockLayout, density->getNTheta(), density->getNPhi());
	advectionCentered<<<gridLayout, blockLayout>>>
	(density->getGPUNextStep(), density->getNextStepPitchInElements());
	density->unbindTexture(&texAdvDensity);

	density->swapGPUBuffer();

	determineLayout(gridLayout, blockLayout, 1, particles->numOfParticles);
	advectionParticles<<<gridLayout, blockLayout>>>
	(particles->coordGPUNextStep, particles->coordGPUThisStep);
	particles->swapGPUBuffers();
# endif

	velPhi->unbindTexture(&texAdvVelPhi);
	velTheta->unbindTexture(&texAdvVelTheta);

	swapVelocityBuffers();
}