# include "../include/KaminoSolver.h"

/*static table2D texProjVelPhi;
static table2D texProjVelTheta;
static table2D texProjPressure;*/

fReal kaminoLerpHost(fReal from, fReal to, fReal alpha)
{
	return (1.0 - alpha) * from + alpha * to;
}

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x);

//nTheta blocks, nPhi threads
__global__ void fillDivergenceKernel
(ComplexFourier* outputF, fReal* velPhi, fReal* velTheta,
	size_t nPhi, size_t velPhiPitchInElements, size_t velThetaPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	int gridPhiId = threadIdx.x;
	int gridThetaId = blockIdx.x;

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
	int thetaNorthIdx = gridThetaId;
	int thetaSouthIdx = gridThetaId + 1;

	uWest = velPhi[gridThetaId * velPhiPitchInElements + phiIdWest];
	uEast = velPhi[gridThetaId * velPhiPitchInElements + phiIdEast];
	vNorth = velTheta[thetaNorthIdx * velThetaPitchInElements + gridPhiId];
	vSouth = velTheta[thetaSouthIdx * velThetaPitchInElements + gridPhiId];

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

//nTheta blocks, n threads
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

//nTheta blocks, n threads
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

//1 block, nTheta threads
__global__ void cacheZeroComponents
(fReal* zeroComponentCache, ComplexFourier* input,
	size_t nPhi)
{
	int thetaIdx = threadIdx.x;
	zeroComponentCache[thetaIdx] = input[thetaIdx * nPhi + nPhi / 2].x;
}

//nTheta blocks, nPhi threads
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

//nTheta - 1 blocks, nPhi threads
__global__ void applyPressureTheta
(fReal* output, fReal* prev, fReal* pressure,
	size_t nPitchInElementsPressure, size_t nPitchInElementsVTheta,
	fReal gridLen)
{
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x + 1;

	int pressureThetaNorthId = thetaId - 1;
	int pressureThetaSouthId = thetaId;
	fReal pressureNorth = pressure[pressureThetaNorthId * nPitchInElementsPressure + phiId];
	fReal pressureSouth = pressure[pressureThetaSouthId * nPitchInElementsPressure + phiId];

	fReal deltaVTheta = (pressureSouth - pressureNorth) / (-gridLen);
	fReal previousVTheta = prev[thetaId * nPitchInElementsVTheta + phiId];
	output[thetaId * nPitchInElementsVTheta + phiId] = previousVTheta + deltaVTheta;
}

//nTheta blocks, nPhi threads
__global__ void applyPressurePhi
(fReal* output, fReal* prev, fReal* pressure,
	size_t nPitchInElementsPressure, size_t nPhi, size_t nPitchInElementsVPhi,
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
		this->nPhi, this->velPhi->getThisStepPitch() / sizeof(fReal), this->velTheta->getThisStepPitch() / sizeof(fReal),
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

	gridLayout = dim3(nTheta - 1);
	blockLayout = dim3(nPhi);
	applyPressureTheta<<<gridLayout, blockLayout>>>
		(velTheta->getGPUNextStep(), velTheta->getGPUThisStep(), pressure->getGPUThisStep(),
			pressure->getThisStepPitch() / sizeof(fReal), velTheta->getNextStepPitch() / sizeof(fReal),
			gridLen);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gridLayout = dim3(nTheta);
	blockLayout = dim3(nPhi);
	applyPressurePhi<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), pressure->getGPUThisStep(),
		pressure->getThisStepPitch() / sizeof(fReal), nPhi, velPhi->getNextStepPitch() / sizeof(fReal),
		gridLen);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	swapAttrBuffers();
}

void KaminoSolver::solveForPolarVelocities()
{
	fReal* u = velPhi->getCPUBuffer();
	fReal* v = velTheta->getCPUBuffer();

	size_t northernBelt = 0;
	size_t northernBeltp1 = northernBelt + 1;
	size_t southernBelt = nTheta - 1; // uTheta->getNTheta() - 2
	size_t southernBeltm1 = southernBelt - 1;
	size_t northernPinch = 0;
	size_t southernPinch = nTheta;

	fReal uNorthP[2] = { 0.0, 0.0 };
	fReal uSouthP[2] = { 0.0, 0.0 };
	static enum coordXY {x, y};

	for (size_t gridPhi = 0; gridPhi < nPhi; ++gridPhi)
	{
		fReal phi = (M_2PI / nPhi) * gridPhi;

		size_t gridPhiP1 = (gridPhi + 1) % nPhi;
		fReal ootBeltUPhi = kaminoLerpHost(u[northernBelt * nPhi + gridPhi], u[northernBelt * nPhi + gridPhiP1], 0.5);
		fReal totBeltUPhi = kaminoLerpHost(u[northernBeltp1 * nPhi + gridPhi], u[northernBeltp1 * nPhi + gridPhiP1], 0.5);
		fReal uPhiLatLine = kaminoLerpHost(ootBeltUPhi, totBeltUPhi, 0.5);
		fReal uThetaLatLine = v[(northernPinch + 1) * nPhi + gridPhi];

		uNorthP[x] += uThetaLatLine * std::cos(phi) - uPhiLatLine * std::sin(phi);
		uNorthP[y] += uThetaLatLine * std::sin(phi) + uPhiLatLine * std::cos(phi);


		ootBeltUPhi = kaminoLerpHost(u[southernBelt * nPhi + gridPhi], u[southernBelt * nPhi + gridPhiP1], 0.5);
		totBeltUPhi = kaminoLerpHost(u[southernBeltm1 * nPhi + gridPhi], u[southernBeltm1 * nPhi + gridPhiP1], 0.5);
		uPhiLatLine = kaminoLerpHost(ootBeltUPhi, totBeltUPhi, 0.5);
		uThetaLatLine = v[(southernPinch - 1) * nPhi + gridPhi];

		uSouthP[x] += -uThetaLatLine * std::cos(phi) - uPhiLatLine * std::sin(phi);
		uSouthP[y] += -uThetaLatLine * std::sin(phi) + uPhiLatLine * std::cos(phi);
	}
	for (unsigned i = 0; i < 2; ++i)
	{
		uNorthP[i] /= nPhi;
		uSouthP[i] /= nPhi;
	}
	//Now we have the projected x, y components at polars
	for (size_t gridPhi = 0; gridPhi < nPhi; ++gridPhi)
	{
		fReal phi = (M_2PI / nPhi) * gridPhi;
		fReal northernUTheta = uNorthP[x] * std::cos(phi) + uNorthP[y] * std::sin(phi);
		v[northernPinch * nPhi + gridPhi] = northernUTheta;
		fReal southernUTheta = -uSouthP[x] * std::cos(phi) - uSouthP[y] * std::sin(phi);
		v[southernPinch * nPhi + gridPhi] = southernUTheta;
	}
	velTheta->copyToGPU();
}