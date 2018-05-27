# include "KaminoSolver.h"

static table2D texAdvVelPhi;
static table2D texAdvVelTheta;

__global__ void advectionVPhiKernel
	(fReal* attributeOutput,
	size_t nTheta, size_t nPhi, size_t nPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
    int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLen;
	
	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhi;
	fReal gThetaTex = (fReal)thetaId / nTheta;

	// Sample the speed
	fReal guPhi = tex2D<fReal>(texAdvVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D<fReal>(texAdvVelTheta, gPhiTex, gThetaTex);

	fReal latRadius = radius * sinf(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal midPhiTex = (midPhi - vPhiPhiOffset * gridLen) / vPhiPhiNorm;
	fReal midThetaTex = (midTheta - vPhiThetaOffset * gridLen) / vPhiThetaNorm;

	fReal muPhi = tex2D<fReal>(texAdvVelPhi, midPhiTex, midThetaTex);
	fReal muTheta = tex2D<fReal>(texAdvVelTheta, midPhiTex, midThetaTex);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;
	fReal pPhiTex = (pPhi - vPhiPhiOffset * gridLen) / vPhiPhiNorm;
	fReal pThetaTex = (pTheta - vPhiThetaOffset * gridLen) / vPhiThetaNorm;

	fReal advectedVal = tex2D<fReal>(texAdvVelPhi, pPhiTex, pThetaTex);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
};

__global__ void advectionVThetaKernel
(fReal* attributeOutput,
	size_t nTheta, size_t nPhi, size_t nPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLen;

	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhi;
	fReal gThetaTex = (fReal)thetaId / nTheta;

	// Sample the speed
	fReal guPhi = tex2D<fReal>(texAdvVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D<fReal>(texAdvVelTheta, gPhiTex, gThetaTex);

	fReal latRadius = radius * sinf(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	// Traced halfway in phi-theta space
	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;
	fReal midPhiTex = (midPhi - vThetaPhiOffset * gridLen) / vThetaPhiNorm;
	fReal midThetaTex = (midTheta - vThetaThetaOffset * gridLen) / vThetaThetaNorm;

	fReal muPhi = tex2D(texAdvVelPhi, midPhiTex, midThetaTex);
	fReal muTheta = tex2D(texAdvVelTheta, midPhiTex, midThetaTex);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;
	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;
	fReal pPhiTex = (pPhi - vThetaPhiOffset * gridLen) / vThetaPhiNorm;
	fReal pThetaTex = (pTheta - vThetaThetaOffset * gridLen) / vThetaThetaNorm;

	fReal advectedVal = tex2D(texAdvVelTheta, pPhiTex, pThetaTex);

	attributeOutput[thetaId * nPitchInElements + phiId] = advectedVal;
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
	(velPhi->getGPUNextStep(), velPhi->getNTheta(), velPhi->getNPhi(), velPhi->getNextStepPitch() / sizeof(fReal),
	gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Advect Theta
	
	gridLayout = dim3(velTheta->getNTheta());
	blockLayout = dim3(velTheta->getNPhi());
	advectionVThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velTheta->getNTheta(), velTheta->getNPhi(), velTheta->getNextStepPitch() / sizeof(fReal),
	gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	velPhi->unbindTexture(&texAdvVelPhi);
	velTheta->unbindTexture(&texAdvVelTheta);

	swapAttrBuffers();
}