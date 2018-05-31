# include "../include/KaminoSolver.cuh"

static table2D texProjVelPhi;
static table2D texProjVelTheta;
static table2D texProjPressure;

static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal gridLenGlobal;

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x);

__global__ void fillDivergenceKernel
(ComplexFourier* outputF, fReal* velPhi, fReal* velTheta,
	size_t velPhiPitchInElements, size_t velThetaPitchInElements)
{
	int gridPhiId = threadIdx.x;
	int gridThetaId = blockIdx.x;
	//fReal gridPhiCoord = ((fReal)gridPhiId + centeredPhiOffset) * gridLen;
	fReal gridThetaCoord = ((fReal)gridThetaId + centeredThetaOffset) * gridLenGlobal;

	fReal uEast = 0.0;
	fReal uWest = 0.0;
	fReal vNorth = 0.0;
	fReal vSouth = 0.0;

	fReal halfStep = 0.5 * gridLenGlobal;
	
	fReal thetaSouth = gridThetaCoord + halfStep;
	fReal thetaNorth = gridThetaCoord - halfStep;

	int phiIdWest = gridPhiId;
	int phiIdEast = (phiIdWest + 1) % nPhiGlobal;

	uWest = velPhi[gridThetaId * velPhiPitchInElements + phiIdWest];
	uEast = velPhi[gridThetaId * velPhiPitchInElements + phiIdEast];

	if (gridThetaId != 0)
	{
		int thetaNorthIdx = gridThetaId - 1;
		vNorth = velTheta[thetaNorthIdx * velThetaPitchInElements + gridPhiId];
	}
	if (gridThetaId != nThetaGlobal - 1)
	{
		int thetaSouthIdx = gridThetaId;
		vSouth = velTheta[thetaSouthIdx * velThetaPitchInElements + gridPhiId];
	}

	fReal invGridSine = 1.0 / sinf(gridThetaCoord);
	fReal sinNorth = sinf(thetaNorth);
	fReal sinSouth = sinf(thetaSouth);
	fReal factor = invGridSine / gridLenGlobal;
	fReal termTheta = factor * (vSouth * sinSouth - vNorth * sinNorth);
	fReal termPhi = factor * (uEast - uWest);

	fReal div = termTheta + termPhi;

	ComplexFourier f;
	f.x = div;
	f.y = 0.0;
	outputF[gridThetaId * nPhiGlobal + gridPhiId] = f;
}

__global__ void shiftFKernel
(ComplexFourier* FFourierInput, fReal* FFourierShiftedReal, fReal* FFourierShiftedImag)
{
	int nIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	int fftIndex = nPhiGlobal / 2 - nIdx;
	if (fftIndex < 0)
		fftIndex += nPhiGlobal;
	//FFourierShifted[thetaIdx * nPhi + phiIdx] = FFourierInput[thetaIdx * nPhi + fftIndex];
	fReal real = FFourierInput[thetaIdx * nPhiGlobal + fftIndex].x / (fReal)nPhiGlobal;
	fReal imag = FFourierInput[thetaIdx * nPhiGlobal + fftIndex].y / (fReal)nPhiGlobal;
	FFourierShiftedReal[nIdx * nThetaGlobal + thetaIdx] = real;
	FFourierShiftedImag[nIdx * nThetaGlobal + thetaIdx] = imag;
}

__global__ void copy2UFourier
(ComplexFourier* UFourierOutput, fReal* UFourierReal, fReal* UFourierImag)
{
	int nIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	ComplexFourier u;
	u.x = UFourierReal[nIdx * nThetaGlobal + thetaIdx];
	u.y = UFourierImag[nIdx * nThetaGlobal + thetaIdx];
	UFourierOutput[thetaIdx * nPhiGlobal + nIdx] = u;
}

__global__ void cacheZeroComponents
(fReal* zeroComponentCache, ComplexFourier* input)
{
	int thetaIdx = threadIdx.x;
	zeroComponentCache[thetaIdx] = input[thetaIdx * nPhiGlobal + nPhiGlobal / 2].x;
}

__global__ void shiftUKernel
(ComplexFourier* UFourierInput, fReal* pressure, fReal* zeroComponentCache,
	size_t nPressurePitchInElements)
{
	int phiIdx = threadIdx.x;
	int thetaIdx = blockIdx.x;
	int fftIndex = 0;
	fReal zeroComponent = zeroComponentCache[thetaIdx];
	if (phiIdx != 0)
		fftIndex = nPhiGlobal - phiIdx;
	fReal pressureVal;

	if (phiIdx % 2 == 0)
		pressureVal = UFourierInput[thetaIdx * nPhiGlobal + fftIndex].x - zeroComponent;
	else
		pressureVal = -UFourierInput[thetaIdx * nPhiGlobal + fftIndex].x - zeroComponent;
	
	pressure[thetaIdx * nPressurePitchInElements + phiIdx] = pressureVal;
}

__global__ void applyPressureTheta
(fReal* output, fReal* prev, fReal* pressure,
	size_t nPitchInElementsPressure, size_t nPitchInElementsVTheta)
{
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;

	int pressureThetaNorthId = thetaId;
	int pressureThetaSouthId = thetaId + 1;
	fReal pressureNorth = pressure[pressureThetaNorthId * nPitchInElementsPressure + phiId];
	fReal pressureSouth = pressure[pressureThetaSouthId * nPitchInElementsPressure + phiId];

	fReal deltaVTheta = (pressureSouth - pressureNorth) / (-gridLenGlobal);
	fReal previousVTheta = prev[thetaId * nPitchInElementsVTheta + phiId];
	output[thetaId * nPitchInElementsVTheta + phiId] = previousVTheta + deltaVTheta;
}
__global__ void applyPressurePhi
(fReal* output, fReal* prev, fReal* pressure,
	size_t nPitchInElementsPressure, size_t nPitchInElementsVPhi)
{
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;

	int pressurePhiWestId;
	if (phiId == 0)
		pressurePhiWestId = nPhiGlobal - 1;
	else
		pressurePhiWestId = phiId - 1;
	int pressurePhiEastId = phiId;

	fReal pressureWest = pressure[thetaId * nPitchInElementsPressure + pressurePhiWestId];
	fReal pressureEast = pressure[thetaId * nPitchInElementsPressure + pressurePhiEastId];

	fReal thetaBelt = (thetaId + centeredThetaOffset) * gridLenGlobal;
	fReal deltaVPhi = (pressureEast - pressureWest) / (-gridLenGlobal * sinf(thetaBelt));
	fReal previousVPhi = prev[thetaId * nPitchInElementsVPhi + phiId];
	output[thetaId * nPitchInElementsVPhi + phiId] = previousVPhi + deltaVPhi;
}

void KaminoSolver::projection()
{
	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, &(this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, &(this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, &(this->radius), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, &(this->timeStep), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, &(this->gridLen), sizeof(fReal)));



	dim3 gridLayout(nTheta);
	dim3 blockLayout(nPhi);
	fillDivergenceKernel<<<gridLayout, blockLayout>>>
	(this->gpuFFourier, this->velPhi->getGPUThisStep(), this->velTheta->getGPUThisStep(),
		this->velPhi->getThisStepPitchInElements(), this->velTheta->getThisStepPitchInElements());
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
	(gpuFFourier, gpuFReal, gpuFImag);
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
	(this->gpuUFourier, this->gpuUReal, this->gpuUImag);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	gridLayout = dim3(1);
	blockLayout = dim3(nTheta);
	cacheZeroComponents<<<gridLayout, blockLayout>>>
	(gpuFZeroComponent, gpuUFourier);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors((cudaError_t)cufftExecC2C(this->kaminoPlan,
		this->gpuUFourier, this->gpuUFourier, CUFFT_FORWARD));
	checkCudaErrors(cudaGetLastError());



	gridLayout = dim3(nTheta);
	blockLayout = dim3(nPhi);
	shiftUKernel<<<gridLayout, blockLayout>>>
	(gpuUFourier, pressure->getGPUThisStep(), this->gpuFZeroComponent,
		pressure->getThisStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//pressure->copyBackToCPU();

	gridLayout = dim3(velTheta->getNTheta());
	blockLayout = dim3(velTheta->getNPhi());
	applyPressureTheta<<<gridLayout, blockLayout>>>
		(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), pressure->getGPUThisStep(),
			pressure->getThisStepPitchInElements(), velTheta->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gridLayout = dim3(velPhi->getNTheta());
	blockLayout = dim3(velPhi->getNPhi());
	applyPressurePhi<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), pressure->getGPUThisStep(),
		pressure->getThisStepPitchInElements(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	swapAttrBuffers();
}