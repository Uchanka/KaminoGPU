# include "../include/KaminoSolver.h"

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x);

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

__global__ void shiftFKernel
(ComplexFourier* FFourierInput, fReal* FFourierShiftedReal, fReal* FFourierShiftedImag,
	size_t nTheta, size_t nPhi)
{
	int nIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	int fftIndex = nPhi / 2 - nIdx;
	if (fftIndex < 0)
		fftIndex += nPhi;
	//FFourierShifted[thetaIdx * nPhi + phiIdx] = FFourierInput[thetaIdx * nPhi + fftIndex];
	FFourierShiftedReal[nIdx * nTheta + thetaIdx] = FFourierInput[thetaIdx * nPhi + fftIndex].x;
	FFourierShiftedImag[nIdx * nTheta + thetaIdx] = FFourierInput[thetaIdx * nPhi + fftIndex].y;
}

__global__ void copy2UFourier
(ComplexFourier* UFourierOutput, fReal* UFourierReal, fReal* UFourierImag,
	size_t nTheta, size_t nPhi)
{
	int nIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	UFourierOutput[thetaIdx * nPhi + nIdx].x = UFourierReal[nIdx * nTheta + thetaIdx];
	UFourierOutput[thetaIdx * nPhi + nIdx].y = UFourierImag[nIdx * nTheta + thetaIdx];
}

__global__ void shiftUKernel
(ComplexFourier* FFourierInput, fReal* FFourierShiftedReal, fReal* FFourierShiftedImag,
	size_t nTheta, size_t nPhi)
{

}


void KaminoSolver::projection()
{
	velPhi->bindTexture(texVelPhi);
	velTheta->bindTexture(texVelTheta);

	dim3 gridLayout(nTheta);
	dim3 blockLayout(nPhi);
	fillDivergenceKernel<<<gridLayout, blockLayout>>>
	(gpuFFourier, 
		nTheta, nPhi,
		gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors((cudaError)cufftExecC2C(this->kaminoPlan,
		gpuFFourier, gpuFFourier, CUFFT_INVERSE));
	checkCudaErrors(cudaGetLastError());



	shiftFKernel<<<gridLayout, blockLayout>>>
	(gpuFFourier, gpuFReal, gpuFImag, nTheta, nPhi);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// Now gpuFDivergence stores all the Fn



	gridLayout = dim3(nPhi);
	blockLayout = dim3(nTheta / 2);
	const unsigned sharedMemSize = nTheta * 5 * sizeof(fReal);
	crKernel<<<gridLayout, blockLayout, sharedMemSize>>>
	(this->gpuA, this->gpuB, this->gpuC, this->gpuFReal, this->gpuUReal);
	checkCudaErrors(cudaGetLastError());
	crKernel<<<gridLayout, blockLayout, sharedMemSize>>>
	(this->gpuA, this->gpuB, this->gpuC, this->gpuFImag, this->gpuUImag);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());



	gridLayout = dim3(nTheta);
	blockLayout = dim3(nPhi);
	copy2UFourier<<<gridLayout, blockLayout>>>
	(this->gpuUFourier, this->gpuUReal, this->gpuUImag, nTheta, nPhi);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors((cudaError)cufftExecC2C(this->kaminoPlan,
		gpuUFourier, gpuUFourier, CUFFT_FORWARD));
	checkCudaErrors(cudaGetLastError());
}