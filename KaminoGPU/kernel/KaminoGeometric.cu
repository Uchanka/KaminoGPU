# include "../include/KaminoSolver.cuh"

static table2D texGeoVelPhi;
static table2D texGeoVelTheta;

static __constant__ size_t nPhiGlobalGeo;
static __constant__ size_t nThetaGlobalGeo;
static __constant__ fReal radiusGlobalGeo;
static __constant__ fReal timeStepGlobalGeo;
static __constant__ fReal gridLenGlobalGeo;

__device__ fReal _root3(fReal x)
{
	fReal s = 1.;
	while (x < 1.)
	{
		x *= 8.;
		s *= 0.5;
	}
	while (x > 8.)
	{
		x *= 0.125;
		s *= 2.;
	}
	fReal r = 1.5;
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	r -= 1. / 3. * (r - x / (r * r));
	return r * s;
}

__device__ fReal root3(double x)
{
	if (x > 0)
		return _root3(x);
	else if (x < 0)
		return -_root3(-x);
	else
		return 0.0;
}

#define eps 1e-7f

__device__ fReal solveCubic(fReal a, fReal b, fReal c)
{
	fReal a2 = a * a;
	fReal q = (a2 - 3 * b) / 9.0;
	//q = q >= 0.0 ? q : -q;
	fReal r = (a * (2.0 * a2 - 9.0 * b) + 27.0 * c) / 54.0;
	
	fReal r2 = r * r;
	fReal q3 = q * q * q;
	fReal A, B;
	if (r2 <= (q3 + eps)) 
	{
		double t = r / sqrtf(q3);
		if (t < -1)
			t = -1;
		if (t > 1)
			t = 1;
		t = acosf(t);
		a /= 3.0; 
		q = -2.0 * sqrtf(q);
		return q * cosf(t / 3.0) - a;
	}
	else
	{
		A = -root3(fabsf(r) + sqrtf(r2 - q3));
		if (r < 0) 
			A = -A;

		B = A == 0 ? 0 : B = q / A;

		a /= 3.0;
		return (A + B) - a;
	}
}

//nTheta - 1 by nPhi
__global__ void geometricKernel
(fReal* velPhiOutput, fReal* velThetaOutput, fReal* velPhiInput, fReal* velThetaInput,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobalGeo / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobalGeo;
	// The factor
	fReal factor = timeStepGlobalGeo * cosf(gTheta) / (radiusGlobalGeo * sinf(gTheta));

	fReal uPrev = velPhiInput[phiId + nPitchInElements * (thetaId + 1)];
	fReal vPrev = velThetaInput[phiId + nPitchInElements * thetaId];

	fReal G;
	fReal uNext;
	if (abs(sinf(gTheta)) < eps)
	{
		G = timeStepGlobalGeo * cosf(gTheta) / (radiusGlobalGeo * sinf(gTheta));
		fReal cof = G * G;
		fReal A = 0.0;
		fReal B = (G * vPrev + 1.0) / cof;
		fReal C = -uPrev / cof;

		uNext = solveCubic(A, B, C);
	}
	else
	{
		uNext = uPrev;
	}
	
	fReal vNext = vPrev + G * uNext * uNext;

	velPhiOutput[(thetaId + 1) * nPitchInElements + phiId] = uNext;
	velThetaOutput[thetaId * nPitchInElements + phiId] = vNext;
}

//1 by nPhi
__global__ void copyKernel(fReal* velPhiOutput, fReal* velPhiInput,
	size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobalGeo / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;

	velPhiOutput[phiId] = velPhiInput[phiId];
	velPhiOutput[phiId + (nThetaGlobalGeo - 1) * nPitchInElements] 
		= velPhiInput[phiId + (nThetaGlobalGeo - 1) * nPitchInElements];
}

void KaminoSolver::geometric()
{
	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, nTheta - 1, nPhi);
	geometricKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
		velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	determineLayout(gridLayout, blockLayout, 1, nPhi);
	copyKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	swapVelocityBuffers();
}