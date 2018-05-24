# include "KaminoSolver.h"

__global__ void advectionKernel
	(fReal* attributeOutput,
	size_t nTheta, size_t nPhi, size_t nPitch,
	fReal phiOffset, fReal thetaOffset,
	fReal gridLen, fReal radius, fReal timeStep,
	fReal phiNorm, fReal thetaNorm)
{
	// Index
    int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + phiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + thetaOffset) * gridLen;
	
	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhi;
	fReal gThetaTex = (fReal)thetaId / nTheta;

	// Sample the speed
	fReal guPhi = tex2D(texVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D(texVelTheta, gPhiTex, gThetaTex);

	fReal latRadius = radius * sinf(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal midPhiTex = (midPhi - phiOffset * gridLen) / phiNorm;
	fReal midThetaTex = (midTheta - thetaOffset * gridLen) / thetaNorm;

	fReal muPhi = tex2D(texVelPhi, midPhiTex, midThetaTex);
	fReal muTheta = tex2D(texVelTheta, midPhiTex, midThetaTex);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;
	fReal pPhiTex = (pPhi - phiOffset * gridLen) / phiNorm;
	fReal pThetaTex = (pTheta - thetaOffset * gridLen) / thetaNorm;

	fReal advectedVal = tex2D(texBeingAdvected, pPhiTex, pThetaTex);

	attributeOutput[thetaId * nPitch + phiId] = advectedVal;
};

/*(fReal* attributeOutput,
	size_t nTheta, size_t nPhi, size_t nPitch,
	fReal phiOffset, fReal thetaOffset, fReal gridLen,
	fReal radius, fReal timeStep, fReal phiNorm, fReal thetaNorm);*/

void KaminoSolver::advection()
{
	//bindVelocity2Tex(texVelPhi, texVelTheta);
	velPhi->bindTexture(texVelPhi);
	velTheta->bindTexture(texVelTheta);
	
	///kernel call goes here
	// Advect Phi
	velPhi->bindTexture(texBeingAdvected);
	fReal phiNorm = M_2PI;
	fReal thetaNorm = M_PI;
	dim3 gridLayout = dim3(velPhi->getNTheta());
	dim3 blockLayout = dim3(velPhi->getNPhi());
	advectionKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getNTheta(), velPhi->getNPhi(), velPhi->getNextStepPitch(),
		velPhi->getPhiOffset(), velPhi->getThetaOffset(), gridLen, radius, timeStep, phiNorm, thetaNorm);
	velPhi->unbindTexture(texBeingAdvected);

	// Advect Theta
	velTheta->bindTexture(texBeingAdvected);
	//texBeingAdvected = texVelTheta;
	phiNorm = M_2PI;
	thetaNorm = M_PI - 2 * gridLen;
	gridLayout = dim3(velTheta->getNTheta());
	blockLayout = dim3(velTheta->getNPhi());
	advectionKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velTheta->getNTheta(), velTheta->getNPhi(), velTheta->getNextStepPitch(),
		velTheta->getPhiOffset(), velTheta->getThetaOffset(), gridLen, radius, timeStep, phiNorm, thetaNorm);
	velTheta->unbindTexture(texBeingAdvected);

	velPhi->unbindTexture(texVelPhi);
	velTheta->unbindTexture(texVelTheta);

	swapAttrBuffers();
}