# include "../include/KaminoSolver.h"

__global__ void geometricPhiKernel
(fReal* attributeOutput,
	size_t nTheta, size_t nPhi, size_t nPitch,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLen;
	// The factor
	fReal factor = timeStep * cosf(gTheta) / (radius * sinf(gTheta));

	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhi;
	fReal gThetaTex = (fReal)thetaId / nTheta;

	// Sample the speed
	fReal guPhi = tex2D(texVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D(texVelTheta, gPhiTex, gThetaTex);

	fReal updateduPhi = guPhi - factor * guPhi * gTheta;

	attributeOutput[thetaId * nPitch + phiId] = updateduPhi;
};

__global__ void geometricThetaKernel
(fReal* attributeOutput,
	size_t nTheta, size_t nPhi, size_t nPitch,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLen;
	// The factor
	fReal factor = timeStep * cosf(gTheta) / (radius * sinf(gTheta));

	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhi;
	fReal gThetaTex = (fReal)thetaId / nTheta;

	// Sample the speed
	fReal guPhi = tex2D(texVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D(texVelTheta, gPhiTex, gThetaTex);

	fReal updateduTheta = guTheta + factor * guPhi * guPhi;

	attributeOutput[thetaId * nPitch + phiId] = updateduTheta;
};

void KaminoSolver::geometric()
{
	velPhi->bindTexture(texVelPhi);
	velTheta->bindTexture(texVelTheta);

	dim3 gridLayout = dim3(velPhi->getNTheta());
	dim3 blockLayout = dim3(velPhi->getNPhi());
	geometricPhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getNTheta(),
		velPhi->getNPhi(), velPhi->getNextStepPitch(),
		gridLen, radius, timeStep);
	geometricThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(),
		velTheta->getNTheta(), velTheta->getNPhi(), velTheta->getNextStepPitch(),
		gridLen, radius, timeStep);

	velPhi->unbindTexture(texVelPhi);
	velTheta->unbindTexture(texVelTheta);

	swapAttrBuffers();
}