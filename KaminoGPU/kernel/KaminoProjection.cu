# include "../include/KaminoSolver.h"

static table2D texProjVelPhi;
static table2D texProjVelTheta;
static table2D texProjPressure;

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x);

__global__ void fillDivergenceKernel
(ComplexFourier* outputF,
	size_t nPhi, size_t nTheta,
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
	fReal thetaSouth = gridThetaCoord + halfStep;
	fReal thetaNorth = gridThetaCoord - halfStep;

	// sample the vPhi at gridThetaCoord
	fReal thetaTex = (gridThetaCoord - vPhiThetaOffset * gridLen) / vPhiThetaNorm;
	// sample the vTheta at gridPhiCoord
	fReal phiTex = (gridPhiCoord - vThetaPhiOffset * gridLen) / vThetaPhiNorm;

	fReal phiEastTex = (phiEast - vPhiPhiOffset * gridLen) / vPhiPhiNorm;
	fReal phiWestTex = (phiWest - vPhiPhiOffset * gridLen) / vPhiPhiNorm;

	uEast = tex2D<fReal>(texProjVelPhi, phiEastTex, thetaTex);
	uWest = tex2D<fReal>(texProjVelPhi, phiWestTex, thetaTex);

	if (gridThetaId != 0)
	{
		fReal thetaNorthTex = (thetaNorth - vThetaThetaOffset * gridLen) / vThetaThetaNorm;
		vNorth = tex2D<fReal>(texProjVelTheta, phiTex, thetaNorthTex);
	}
	if (gridThetaId != nTheta - 1)
	{
		fReal thetaSouthTex = (thetaSouth - vThetaThetaOffset * gridLen) / vThetaThetaNorm;
		vSouth = tex2D<fReal>(texProjVelTheta, phiTex, thetaSouthTex);
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
	size_t nPhi, size_t nTheta)
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
	size_t nPhi, size_t nTheta)
{
	int nIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	UFourierOutput[thetaIdx * nPhi + nIdx].x = UFourierReal[nIdx * nTheta + thetaIdx];
	UFourierOutput[thetaIdx * nPhi + nIdx].y = UFourierImag[nIdx * nTheta + thetaIdx];
}

__global__ void shiftUKernel
(ComplexFourier* UFourierInput, fReal* pressure,
	size_t nPhi, size_t nTheta, size_t nPressurePitchInElements)
{
	int phiIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	int fftIndex = 0;
	fReal zeroComponent = UFourierInput[thetaIdx * nPhi + nPhi / 2].x;
	if (phiIdx != 0)
		fftIndex = nPhi - phiIdx;
	fReal pressureVal = 0.0;
	int bit = 0;
	if (phiIdx % 2 == 0)
		bit = 1;
	else
		bit = -1;
	pressureVal = bit * UFourierInput[thetaIdx * nPhi + phiIdx].x - zeroComponent;
	pressure[thetaIdx * nPressurePitchInElements + phiIdx] = pressureVal;
}

__global__ void applyPressureTheta
(fReal* output,
	size_t nPhi, size_t nTheta, size_t nPitchInElements,
	fReal gridLen)
{
	int thetaId = threadIdx.x;
	int phiId = blockIdx.x;

	fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLen;
	fReal thetaSouth = gTheta + 0.5 * gridLen;
	fReal thetaNorth = gTheta - 0.5 * gridLen;

	fReal texPhi = (fReal)phiId / nPhi;
	fReal texTheta = (fReal)thetaId / nTheta;
	fReal texThetaNorth = (thetaNorth - vThetaThetaOffset * gridLen) / pressureThetaNorm;
	fReal texThetaSouth = (thetaSouth - vThetaThetaOffset * gridLen) / pressureThetaNorm;

	fReal previousVTheta = tex2D<fReal>(texProjVelTheta, texPhi, texTheta);
	fReal pressureNorth = tex2D<fReal>(texProjPressure, texPhi, texThetaNorth);
	fReal pressureSouth = tex2D<fReal>(texProjPressure, texPhi, texThetaSouth);

	fReal pressureTheta = pressureSouth - pressureNorth;
	fReal deltaVTheta = -pressureTheta / gridLen;

	output[thetaId * nPhi + phiId] = previousVTheta + deltaVTheta;
}
__global__ void applyPressurePhi
(fReal* output,
	size_t nPhi, size_t nTheta, size_t nPitchInElements,
	fReal gridLen)
{
	int thetaId = threadIdx.x;
	int phiId = blockIdx.x;

	fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLen;
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLen;
	fReal phiEast = gPhi + 0.5 * gridLen;
	fReal phiWest = gPhi - 0.5 * gridLen;

	fReal texPhi = (fReal)phiId / nPhi;
	fReal texTheta = (fReal)thetaId / nTheta;
	fReal texPhiEast = (phiEast - vPhiPhiOffset * gridLen) / pressurePhiNorm;
	fReal texPhiWest = (phiWest - vPhiPhiOffset * gridLen) / pressurePhiNorm;

	fReal previousVPhi = tex2D<fReal>(texProjVelPhi, texPhi, texTheta);
	fReal pressureEast = tex2D<fReal>(texProjPressure, texPhiEast, texTheta);
	fReal pressureWest = tex2D<fReal>(texProjPressure, texPhiWest, texTheta);

	fReal pressurePhi = pressureEast - pressureWest;
	fReal deltaVPhi = -pressurePhi / (gridLen * sinf(gTheta));

	output[thetaId * nPhi + phiId] = previousVPhi + deltaVPhi;
}

void KaminoSolver::projection()
{
	setTextureParams(&texProjVelPhi);
	setTextureParams(&texProjVelTheta);
	setTextureParams(&texProjPressure);
	velPhi->bindTexture(&texProjVelPhi);
	velTheta->bindTexture(&texProjVelTheta);
	pressure->bindTexture(&texProjPressure);

	dim3 gridLayout(nTheta);
	dim3 blockLayout(nPhi);
	fillDivergenceKernel<<<gridLayout, blockLayout>>>
	(gpuFFourier, 
		nPhi, nTheta,
		gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
		this->gpuFFourier, this->gpuFFourier, CUFFT_INVERSE));
	checkCudaErrors(cudaGetLastError());



	shiftFKernel<<<gridLayout, blockLayout>>>
	(gpuFFourier,
		gpuFReal, gpuFImag,
		nPhi, nTheta);
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

	crKernel<<<gridLayout, blockLayout, sharedMemSize>>>
	(this->gpuA, this->gpuB, this->gpuC, this->gpuFImag, this->gpuUImag);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	gridLayout = dim3(nTheta);
	blockLayout = dim3(nPhi);
	copy2UFourier<<<gridLayout, blockLayout>>>
	(this->gpuUFourier,
		this->gpuUReal, this->gpuUImag,
		nPhi, nTheta);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
		this->gpuUFourier, this->gpuUFourier, CUFFT_FORWARD));
	checkCudaErrors(cudaGetLastError());



	shiftUKernel<<<gridLayout, blockLayout>>>
	(gpuUFourier, pressure->getGPUThisStep(),
		nPhi, nTheta, pressure->getThisStepPitch() / sizeof(fReal));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gridLayout = dim3(velTheta->getNTheta());
	blockLayout = dim3(velTheta->getNPhi());
	applyPressureTheta<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(),
		velTheta->getNPhi(), velTheta->getNTheta(), velTheta->getNextStepPitch() / sizeof(fReal),
		gridLen);
	checkCudaErrors(cudaGetLastError());

	gridLayout = dim3(velPhi->getNTheta());
	blockLayout = dim3(velPhi->getNPhi());
	applyPressurePhi<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(),
		velPhi->getNPhi(), velPhi->getNTheta(), velPhi->getNextStepPitch() / sizeof(fReal),
		gridLen);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	swapAttrBuffers();
}