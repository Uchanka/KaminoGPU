# include "../include/KaminoSolver.h"

static table2D texProjVelPhi;
static table2D texProjVelTheta;
static table2D texProjPressure;

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x);

__global__ void fillDivergenceKernel
(ComplexFourier* outputF, fReal* velPhi, fReal* velTheta,
	size_t nPhi, size_t nTheta, size_t velPhiPitchInElements, size_t velThetaPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	int gridPhiId = threadIdx.x;
	int gridThetaId = blockIdx.x;
	//fReal gridPhiCoord = ((fReal)gridPhiId + centeredPhiOffset) * gridLen;
	fReal gridThetaCoord = ((fReal)gridThetaId + centeredThetaOffset) * gridLen;

	fReal uEast = 0.0;
	fReal uWest = 0.0;
	fReal vNorth = 0.0;
	fReal vSouth = 0.0;

	fReal halfStep = 0.5 * gridLen;
	
	fReal thetaSouth = gridThetaCoord + halfStep;
	fReal thetaNorth = gridThetaCoord - halfStep;

	int phiIdWest = gridPhiId;
	int phiIdEast = (phiIdWest + 1) % nPhi;

	uWest = velPhi[gridThetaId * velPhiPitchInElements + phiIdWest];
	uEast = velPhi[gridThetaId * velPhiPitchInElements + phiIdEast];

	if (gridThetaId != 0)
	{
		int thetaNorthIdx = gridThetaId - 1;
		vNorth = velTheta[thetaNorthIdx * velThetaPitchInElements + gridPhiId];
	}
	if (gridThetaId != nTheta - 1)
	{
		int thetaSouthIdx = gridThetaId;
		vSouth = velTheta[thetaSouthIdx * velThetaPitchInElements + gridPhiId];
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
	fReal real = FFourierInput[thetaIdx * nPhi + fftIndex].x / (fReal)nPhi;
	fReal imag = FFourierInput[thetaIdx * nPhi + fftIndex].y / (fReal)nPhi;
	FFourierShiftedReal[nIdx * nTheta + thetaIdx] = real;
	FFourierShiftedImag[nIdx * nTheta + thetaIdx] = imag;
}

__global__ void copy2UFourier
(ComplexFourier* UFourierOutput, fReal* UFourierReal, fReal* UFourierImag,
	size_t nPhi, size_t nTheta)
{
	int nIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	ComplexFourier u;
	u.x = UFourierReal[nIdx * nTheta + thetaIdx];
	u.y = UFourierImag[nIdx * nTheta + thetaIdx];
	UFourierOutput[thetaIdx * nPhi + nIdx] = u;
}

__global__ void cacheZeroComponents
(fReal* zeroComponentCache, ComplexFourier* input,
	size_t nPhi)
{
	int thetaIdx = threadIdx.x;
	zeroComponentCache[thetaIdx] = input[thetaIdx * nPhi + nPhi / 2].x;
}

__global__ void shiftUKernel
(ComplexFourier* UFourierInput, fReal* pressure, fReal* zeroComponentCache,
	size_t nPhi, size_t nTheta, size_t nPressurePitchInElements)
{
	int phiIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	int fftIndex = 0;
	fReal zeroComponent = zeroComponentCache[thetaIdx];
	if (phiIdx != 0)
		fftIndex = nPhi - phiIdx;
	fReal pressureVal;

	if (phiIdx % 2 == 0)
		pressureVal = UFourierInput[thetaIdx * nPhi + fftIndex].x - zeroComponent;
	else
		pressureVal = -UFourierInput[thetaIdx * nPhi + fftIndex].x - zeroComponent;
	
	pressure[thetaIdx * nPressurePitchInElements + phiIdx] = pressureVal;
}

__global__ void applyPressureTheta
(fReal* output, fReal* prev, fReal* pressure, size_t nPitchInElementsPressure,
	size_t nPhi, size_t nTheta, size_t nPitchInElementsVTheta,
	fReal gridLen)
{
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;

	int pressureThetaNorthId = thetaId;
	int pressureThetaSouthId = thetaId + 1;
	fReal pressureNorth = pressure[pressureThetaNorthId * nPitchInElementsPressure + phiId];
	fReal pressureSouth = pressure[pressureThetaSouthId * nPitchInElementsPressure + phiId];

	fReal deltaVTheta = (pressureSouth - pressureNorth) / (-gridLen);
	fReal previousVTheta = prev[thetaId * nPitchInElementsVTheta + phiId];
	output[thetaId * nPitchInElementsVTheta + phiId] = previousVTheta + deltaVTheta;
}

__global__ void applyPressurePhi
(fReal* output, fReal* prev, fReal* pressure, size_t nPitchInElementsPressure,
	size_t nPhi, size_t nTheta, size_t nPitchInElementsVPhi,
	fReal gridLen)
{
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;

	int pressurePhiWestId;
	if (phiId == 0)
		pressurePhiWestId = nPhi - 1;
	else
		pressurePhiWestId = phiId - 1;
	int pressurePhiEastId = phiId;

	fReal pressureWest = pressure[thetaId * nPitchInElementsPressure + pressurePhiWestId];
	fReal pressureEast = pressure[thetaId * nPitchInElementsPressure + pressurePhiEastId];

	fReal thetaBelt = (thetaId + centeredThetaOffset) * gridLen;
	fReal deltaVPhi = (pressureEast - pressureWest) / (-gridLen * sinf(thetaBelt));
	fReal previousVPhi = prev[thetaId * nPitchInElementsVPhi + phiId];
	output[thetaId * nPitchInElementsVPhi + phiId] = previousVPhi + deltaVPhi;
}

void KaminoSolver::projection()
{
	dim3 gridLayout(nTheta);
	dim3 blockLayout(nPhi);
	fillDivergenceKernel<<<gridLayout, blockLayout>>>
	(this->gpuFFourier, this->velPhi->getGPUThisStep(), this->velTheta->getGPUThisStep(),
		this->nPhi, this->nTheta, this->velPhi->getThisStepPitch() / sizeof(fReal), this->velTheta->getThisStepPitch() / sizeof(fReal),
		this->gridLen, this->radius, this->timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	// Note that cuFFT inverse returns results are SigLen times larger
	checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
		this->gpuFFourier, this->gpuFFourier, CUFFT_INVERSE));
	checkCudaErrors(cudaGetLastError());



	// Siglen is nPhi
	gridLayout = dim3(nTheta);
	blockLayout = dim3(nPhi);
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

	gridLayout = dim3(nPhi);
	blockLayout = dim3(nTheta / 2);
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



	gridLayout = dim3(1);
	blockLayout = dim3(nTheta);
	cacheZeroComponents<<<gridLayout, blockLayout>>>
	(gpuFZeroComponent, gpuUFourier, nPhi);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
		this->gpuUFourier, this->gpuUFourier, CUFFT_FORWARD));
	checkCudaErrors(cudaGetLastError());



	gridLayout = dim3(nTheta);
	blockLayout = dim3(nPhi);
	shiftUKernel<<<gridLayout, blockLayout>>>
	(gpuUFourier, pressure->getGPUThisStep(), this->gpuFZeroComponent,
		nPhi, nTheta, pressure->getThisStepPitch() / sizeof(fReal));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	pressure->copyBackToCPU();

	gridLayout = dim3(velTheta->getNTheta() - 1);
	blockLayout = dim3(velTheta->getNPhi());
	applyPressureTheta<<<gridLayout, blockLayout>>>
		(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), pressure->getGPUThisStep(), pressure->getThisStepPitch() / sizeof(fReal),
		velTheta->getNPhi(), velTheta->getNTheta(), velTheta->getNextStepPitch() / sizeof(fReal),
		gridLen);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gridLayout = dim3(velPhi->getNTheta());
	blockLayout = dim3(velPhi->getNPhi());
	applyPressurePhi<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), pressure->getGPUThisStep(), pressure->getThisStepPitch() / sizeof(fReal),
		velPhi->getNPhi(), velPhi->getNTheta(), velPhi->getNextStepPitch() / sizeof(fReal),
		gridLen);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	swapAttrBuffers();
}