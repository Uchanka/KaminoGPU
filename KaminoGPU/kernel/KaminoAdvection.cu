# include "KaminoSolver.h"

static table2D texAdvVelPhi;
static table2D texAdvVelTheta;

__device__ fReal kaminoLerp(fReal from, fReal to, fReal alpha)
{
	return (1.0 - alpha) * from + alpha * to;
}

__device__ fReal sampleAt
(fReal* attribute, fReal phi, fReal theta,
	fReal phiOffset, fReal thetaOffset, fReal gridLen,
	size_t nTheta, size_t nPhi, size_t nPitch)
{
	fReal phi_0ed = phi - gridLen * phiOffset;
	fReal theta_0ed = theta - gridLen * thetaOffset;
	fReal invGridLen = 1.0 / gridLen;

	if (phi_0ed < 0)
		phi_0ed = M_2PI - phi_0ed;
	if (phi_0ed >= M_2PI)
		phi_0ed = phi_0ed - M_2PI;
	if (theta_0ed < 0)
		theta_0ed = -theta_0ed;
	if (theta_0ed > M_PI)
		theta_0ed = M_2PI - theta_0ed;

	phi_0ed = phi_0ed * invGridLen;
	theta_0ed = theta_0ed * invGridLen;

	int phiIndex = floorf(phi_0ed);
	int thetaIndex = floorf(theta_0ed);
	fReal alphaPhi = phi_0ed - phiIndex;
	fReal alphaTheta = theta_0ed - thetaIndex;

	int phiP1Index = (phiIndex + 1) % nPhi;
	int thetaP1Index = (thetaIndex + 1) % nTheta;

	fReal Nwest = attribute[thetaIndex * nPitch + phiIndex];
	fReal Neast = attribute[thetaIndex * nPitch + phiP1Index];
	fReal Swest = attribute[thetaP1Index * nPitch + phiIndex];
	fReal Seast = attribute[thetaP1Index * nPitch + phiP1Index];

	fReal northernBelt = kaminoLerp(Nwest, Neast, alphaPhi);
	fReal southernBelt = kaminoLerp(Swest, Seast, alphaPhi);
	fReal sampledVal = kaminoLerp(northernBelt, southernBelt, alphaTheta);

	return sampledVal;
}

__global__ void advectionVPhiKernel
	(fReal* attributeOutput, fReal* inputVPhi, fReal* inputVTheta,
	size_t nPhi, size_t nTheta, size_t nPitch,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
    int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLen;
	
	// Sample the speed
	fReal guPhi = sampleAt(inputVPhi, gPhi, gTheta, vPhiPhiOffset, vPhiThetaOffset,
		gridLen, nTheta, nPhi, nPitch);
	fReal guTheta = sampleAt(inputVTheta, gPhi, gTheta, vThetaPhiOffset, vThetaThetaOffset,
		gridLen, nTheta - 1, nPhi, nPitch);

	fReal latRadius = radius * sinf(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	
	fReal muPhi = sampleAt(inputVPhi, midPhi, midTheta, vPhiPhiOffset, vPhiThetaOffset,
		gridLen, nTheta, nPhi, nPitch);
	fReal muTheta = sampleAt(inputVTheta, midPhi, midTheta, vThetaPhiOffset, vThetaThetaOffset,
		gridLen, nTheta - 1, nPhi, nPitch);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;

	fReal advectedVal = sampleAt(inputVPhi, pPhi, pTheta, vPhiPhiOffset, vThetaThetaOffset,
		gridLen, nTheta, nPhi, nPitch);

	attributeOutput[thetaId * nPitch + phiId] = advectedVal;
};

__global__ void advectionVThetaKernel
(fReal* attributeOutput, fReal* inputVPhi, fReal* inputVTheta,
	size_t nPhi, size_t nTheta, size_t nPitch,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLen;

	// Sample the speed
	fReal guPhi = sampleAt(inputVPhi, gPhi, gTheta, vPhiPhiOffset, vPhiThetaOffset,
		gridLen, nTheta + 1, nPhi, nPitch);
	fReal guTheta = sampleAt(inputVTheta, gPhi, gTheta, vThetaPhiOffset, vThetaThetaOffset,
		gridLen, nTheta, nPhi, nPitch);

	fReal latRadius = radius * sinf(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	
	fReal muPhi = sampleAt(inputVPhi, midPhi, midTheta, vPhiPhiOffset, vPhiThetaOffset,
		gridLen, nTheta + 1, nPhi, nPitch);
	fReal muTheta = sampleAt(inputVTheta, midPhi, midTheta, vThetaPhiOffset, vThetaThetaOffset,
		gridLen, nTheta, nPhi, nPitch);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;
	
	fReal advectedVal = sampleAt(inputVTheta, pPhi, pTheta, vThetaPhiOffset, vThetaThetaOffset,
		gridLen, nTheta, nPhi, nPitch);

	attributeOutput[thetaId * nPitch + phiId] = advectedVal;
};

void KaminoSolver::advection()
{
	//bindVelocity2Tex(texVelPhi, texVelTheta);
	setTextureParams(&texAdvVelPhi);
	setTextureParams(&texAdvVelTheta);
	velPhi->bindTexture(&texAdvVelPhi);
	velTheta->bindTexture(&texAdvVelTheta);


	
	///kernel call goes here
	// Advect Phi
	dim3 gridLayout = dim3(velPhi->getNTheta());
	dim3 blockLayout = dim3(velPhi->getNPhi());
	advectionVPhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velPhi->getNPhi(), velPhi->getNTheta(),
		velPhi->getNextStepPitch(), gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Advect Theta
	
	gridLayout = dim3(velTheta->getNTheta());
	blockLayout = dim3(velTheta->getNPhi());
	advectionVThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(), velTheta->getNPhi(), velTheta->getNTheta(),
		velTheta->getNextStepPitch(), gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	velPhi->unbindTexture(&texAdvVelPhi);
	velTheta->unbindTexture(&texAdvVelTheta);

	swapAttrBuffers();
}