# include "KaminoSolver.h"

/*static table2D texAdvVelPhi;
static table2D texAdvVelTheta;*/

__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha)
{
	return (1.0 - alpha) * from + alpha * to;
}

// Phi: 0 - 2pi  Theta: 0 - pi
__device__ void validatePhiTheta(fReal &phi, fReal &theta)
{
	int loops = (int)(floorf(theta / M_2PI));
	theta = theta - loops * M_2PI;
	// Now theta is in 0-2pi range
	if (theta > M_PI)
	{
		theta = M_2PI - theta;
		phi += M_PI;
	}
	loops = (int)(floorf(phi / M_2PI));
	phi = phi - loops * M_2PI;
}

__device__ fReal sampleVPhiAt(fReal* vPhi, fReal rawPhi, fReal rawTheta,
	size_t nPhi, size_t nTheta, size_t nPitchInElements,
	fReal gridLen)
{
	fReal phi = rawPhi - gridLen * vPhiPhiOffset;
	fReal theta = rawTheta - gridLen * vPhiThetaOffset;

	validatePhiTheta(phi, theta);

	fReal normedPhi = phi / gridLen;
	fReal normedTheta = theta / gridLen;

	int phiIndex = (int)floorf(normedPhi);
	int thetaIndex = (int)floorf(normedTheta);

	fReal alphaPhi = normedPhi - (fReal)(phiIndex);
	fReal alphaTheta = normedTheta - (fReal)(thetaIndex);

	int phiLower = phiIndex % nPhi;
	int phiHigher = (phiIndex + 1) % nPhi;
	int thetaLower = thetaIndex;
	int thetaHigher = thetaIndex + 1 >= nTheta ? thetaIndex : thetaIndex + 1;

	fReal NW = vPhi[thetaLower * nPitchInElements + phiLower];
	fReal NE = vPhi[thetaLower * nPitchInElements + phiHigher];

	fReal SW = vPhi[thetaHigher * nPitchInElements + phiLower];
	fReal SE = vPhi[thetaHigher * nPitchInElements + phiHigher];

	fReal lowerBelt = kaminoLerp(NW, NE, alphaPhi);
	fReal higherBelt = kaminoLerp(SW, SE, alphaPhi);
	return kaminoLerp(lowerBelt, higherBelt, alphaTheta);
}

__device__ fReal sampleVThetaAt(fReal* vTheta, fReal rawPhi, fReal rawTheta,
	size_t nPhi, size_t nTheta, size_t nPitchInElements,
	fReal gridLen)
{
	fReal phi = rawPhi - gridLen * vThetaPhiOffset;
	fReal theta = rawTheta - gridLen * vThetaThetaOffset;

	validatePhiTheta(phi, theta);

	fReal normedPhi = phi / gridLen;
	fReal normedTheta = theta / gridLen;

	int phiIndex = (int)floorf(normedPhi);
	int thetaIndex = (int)floorf(normedTheta);

	fReal alphaPhi = normedPhi - (fReal)(phiIndex);
	fReal alphaTheta = normedTheta - (fReal)(thetaIndex);

	int phiLower = phiIndex % nPhi;
	int phiHigher = (phiIndex + 1) % nPhi;
	int thetaLower = thetaIndex;
	int thetaHigher = thetaIndex + 1 >= nTheta ? thetaIndex : thetaIndex + 1;

	fReal NW = vTheta[thetaLower * nPitchInElements + phiLower];
	fReal NE = vTheta[thetaLower * nPitchInElements + phiHigher];

	fReal SW = vTheta[thetaHigher * nPitchInElements + phiLower];
	fReal SE = vTheta[thetaHigher * nPitchInElements + phiHigher];

	fReal lowerBelt = kaminoLerp(NW, NE, alphaPhi);
	fReal higherBelt = kaminoLerp(SW, SE, alphaPhi);
	return kaminoLerp(lowerBelt, higherBelt, alphaTheta);
}

//nTheta blocks, nPhi threads
__global__ void advectionUPhiKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta,
	size_t nPhi, size_t nTheta, size_t nPhiPitchInElements, size_t nThetaPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLen;

	fReal latRadius = radius * sinf(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal guPhi = sampleVPhiAt(velPhi, gPhi, gTheta,
		nPhi, nTheta, nPhiPitchInElements, gridLen);
	fReal gvTheta = sampleVThetaAt(velTheta, gPhi, gTheta,
		nPhi, nTheta - 1, nThetaPitchInElements, gridLen);

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = gvTheta * cofTheta;

	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;

	fReal muPhi = sampleVPhiAt(velPhi, midPhi, midTheta,
		nPhi, nTheta, nPhiPitchInElements, gridLen);
	fReal mvTheta = sampleVThetaAt(velTheta, midPhi, midTheta,
		nPhi, nTheta + 1, nThetaPitchInElements, gridLen);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (mvTheta + gvTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;

	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;

	fReal advectedVal = sampleVPhiAt(velPhi, pPhi, pTheta,
		nPhi, nTheta, nPhiPitchInElements, gridLen);

	attributeOutput[thetaId * nPhiPitchInElements + phiId] = advectedVal;
};

//nTheta - 1 blocks, nPhi threads
__global__ void advectionVThetaKernel
(fReal* attributeOutput, fReal* velPhi, fReal* velTheta,
	size_t nPhi, size_t nTheta, size_t nPhiPitchInElements, size_t nThetaPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLen;

	fReal latRadius = radius * sinf(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal guPhi = sampleVPhiAt(velPhi, gPhi, gTheta,
		nPhi, nTheta, nPhiPitchInElements, gridLen);
	fReal gvTheta = sampleVThetaAt(velTheta, gPhi, gTheta,
		nPhi, nTheta - 1, nThetaPitchInElements, gridLen);

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = gvTheta * cofTheta;

	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;

	fReal muPhi = sampleVPhiAt(velPhi, midPhi, midTheta,
		nPhi, nTheta, nPhiPitchInElements, gridLen);
	fReal mvTheta = sampleVThetaAt(velTheta, midPhi, midTheta,
		nPhi, nTheta + 1, nThetaPitchInElements, gridLen);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (mvTheta + gvTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;

	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;

	fReal advectedVal = sampleVThetaAt(velPhi, pPhi, pTheta,
		nPhi, nTheta + 1, nThetaPitchInElements, gridLen);

	attributeOutput[thetaId * nThetaPitchInElements + phiId] = advectedVal;
};

void KaminoSolver::advection()
{
	// Advect Phi
	dim3 gridLayout = dim3(nTheta);
	dim3 blockLayout = dim3(nPhi);
	advectionUPhiKernel<<<gridLayout, blockLayout>>>
		(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
			nPhi, nTheta, velPhi->getNextStepPitch() / sizeof(fReal), velTheta->getNextStepPitch() / sizeof(fReal),
			gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Advect Theta

	gridLayout = dim3(nTheta - 1);
	blockLayout = dim3(nPhi);
	advectionVThetaKernel<<<gridLayout, blockLayout>>>
		(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
			nPhi, nTheta, velPhi->getNextStepPitch() / sizeof(fReal), velTheta->getNextStepPitch() / sizeof(fReal),
			gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	swapAttrBuffers();
}