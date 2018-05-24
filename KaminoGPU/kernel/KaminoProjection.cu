# include "../include/KaminoSolver.h"

__global__ void fillDivergenceKernel
(ComplexFourier* outputF,
	size_t nTheta, size_t nPhi,
	fReal gridLen, fReal radius, fReal timeStep)
{
	int gridPhiId = threadIdx.x;
	int gridThetaId = blockIdx.x;
	fReal gridPhiCoord = ((fReal)gridPhiId + centeredPhiOffset) * gridLen;
	fReal gridThetaCoord = ((fReal)gridThetaId + centeredThetaOffset) * gridLen;

	fReal uEast = 0.0;
	fReal uWest = 0.0;
	fReal vNorth = 0.0;
	fReal vSouth = 0.0;

	fReal halfStep = 0.5 * gridLen;
	fReal phiEast = gridPhiCoord + halfStep;
	fReal phiWest = gridPhiCoord - halfStep;
	fReal thetaNorth = gridThetaCoord - halfStep;
	fReal thetaSouth = gridThetaCoord + halfStep;

	// sample the vPhi at gridThetaCoord
	fReal thetaTex = (gridThetaCoord - vPhiThetaOffset * gridLen) / vPhiThetaNorm;
	// sample the vTheta at gridPhiCoord
	fReal phiTex = (gridPhiCoord - vThetaPhiOffset * gridLen) / vThetaPhiNorm;

	fReal phiEastTex = (phiEast - vPhiPhiOffset * gridLen) / vPhiPhiNorm;
	fReal phiWestTex = (phiWest - vPhiPhiOffset * gridLen) / vPhiPhiNorm;

	uEast = tex2D(texVelPhi, phiEastTex, thetaTex);
	uWest = tex2D(texVelPhi, phiWestTex, thetaTex);

	if (gridThetaId != 0)
	{
		fReal thetaNorthTex = (thetaNorth - vThetaThetaOffset * gridLen) / vThetaThetaNorm;
		vNorth = tex2D(texVelTheta, phiTex, thetaNorthTex);
	}
	if (gridThetaId != nTheta - 1)
	{
		fReal thetaSouthTex = (thetaSouth - vThetaThetaOffset * gridLen) / vThetaThetaNorm;
		vSouth = tex2D(texVelTheta, phiTex, thetaSouthTex);
	}

	fReal invGridSine = 1.0 / sinf(gridThetaCoord);
	fReal sinNorth = sinf(thetaNorth);
	fReal sinSouth = sinf(thetaSouth);
	fReal factor = invGridSine / gridLen;
	fReal termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
	fReal termPhi = factor * (uEast - uWest);

	fReal div = termTheta + termPhi;

	ComplexFourier f;
	f.x = div;
	f.y = 0.0;
	outputF[gridThetaId * nPhi + gridPhiId] = f;
}

__global__ void shiftF
(ComplexFourier* input, ComplexFourier*output, size_t nTheta, size_t nPhi)
{

}

void KaminoSolver::projection()
{
	velPhi->bindTexture(texVelPhi);
	velTheta->bindTexture(texVelTheta);

	dim3 gridLayout(nTheta);
	dim3 blockLayout(nPhi);
	fillDivergenceKernel<<<gridLayout, blockLayout>>>
	(gpuFPool, 
		nTheta, nPhi,
		gridLen, radius, timeStep);

	checkCudaErrors(cufftExecC2C(this->kaminoPlan,
		gpuFDivergence, gpuFFourier, CUFFT_INVERSE));
}