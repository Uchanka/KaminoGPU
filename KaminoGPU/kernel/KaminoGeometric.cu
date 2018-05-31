# include "../include/KaminoSolver.h"

/*static table2D texGeoVelPhi;
static table2D texGeoVelTheta;*/

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

//nTheta blocks, nPhi threads
__global__ void geometricPhiKernel
(fReal* velPhiOutput, fReal* velPhiInput, fReal* velThetaInput,
	size_t nPhiPitchInElements, size_t nThetaPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	int thetaIndex = blockIdx.x;
	int phiIndex = threadIdx.x;

	fReal thetaValue = ((fReal)thetaIndex + vPhiThetaOffset) * gridLen;

	int uPhiIndex = thetaIndex * nPhiPitchInElements + phiIndex;
	int vThetaIndex = thetaIndex * nThetaPitchInElements + phiIndex;
	fReal uPhiPrev = velPhiInput[uPhiIndex];
	fReal vThetaPrev = 0.5 * (velThetaInput[vThetaIndex]
		+ velThetaInput[vThetaIndex + nThetaPitchInElements]);

	fReal G = timeStep * cosf(thetaValue) / (radius * sinf(thetaValue));
	fReal coef = 1.0 / (G * G);
	fReal A = 0.0;
	fReal B = (G * vThetaPrev + 1.0) * coef;
	fReal C = -uPhiPrev * coef;

	fReal uPhiNext = solveCubic(A, B, C);

	velPhiOutput[uPhiIndex] = uPhiNext;
}

//nTheta - 1 blocks, nPhi threads
__global__ void geometricThetaKernel
(fReal* velThetaOutput, fReal* velPhiInput, fReal* velThetaInput,
	size_t nPhiPitchInElements, size_t nThetaPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	int thetaIndex = blockIdx.x + 1;
	int phiIndex = threadIdx.x;

	fReal thetaValue = ((fReal)thetaIndex + vThetaThetaOffset) * gridLen;

	int uPhiIndex = thetaIndex * nPhiPitchInElements + phiIndex;
	int vThetaIndex = thetaIndex * nThetaPitchInElements + phiIndex;
	fReal vThetaPrev = velThetaInput[vThetaIndex];
	fReal uPhiNext = velPhiInput[uPhiIndex];

	fReal G = timeStep * cosf(thetaValue) / (radius * sinf(thetaValue));

	fReal vThetaNext = vThetaPrev + G * uPhiNext * uPhiNext;

	velThetaOutput[vThetaIndex] = vThetaNext;
}

/*__global__ void geometricPhiKernel
(fReal* velPhiOutput, fReal* velPhiInput,
	size_t nPhi, size_t nTheta, size_t nPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLen;
	// The factor
	fReal factor = timeStep * cosf(gTheta) / (radius * sinf(gTheta));

	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhi;
	fReal gThetaTex = (fReal)thetaId / nTheta;

	// Sample the speed
	fReal guPhi = tex2D(texGeoVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D(texGeoVelTheta, gPhiTex, gThetaTex);

	fReal updateduPhi = guPhi - factor * guPhi * gTheta;

	attributeOutput[thetaId * nPitchInElements + phiId] = updateduPhi;
};

__global__ void geometricThetaKernel
(fReal* velThetaOutput, fReal* velThetaInput,
	size_t nPhi, size_t nTheta, size_t nPitchInElements,
	fReal gridLen, fReal radius, fReal timeStep)
{
	// Index
	int phiId = threadIdx.x;
	int thetaId = blockIdx.x;
	// Coord in phi-theta space
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLen;
	// The factor
	fReal factor = timeStep * cosf(gTheta) / (radius * sinf(gTheta));

	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhi;
	fReal gThetaTex = (fReal)thetaId / nTheta;

	// Sample the speed
	fReal guPhi = tex2D(texGeoVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D(texGeoVelTheta, gPhiTex, gThetaTex);

	fReal updateduTheta = guTheta + factor * guPhi * guPhi;

	attributeOutput[thetaId * nPitchInElements + phiId] = updateduTheta;
};*/

void KaminoSolver::geometric()
{
	dim3 gridLayout = dim3(nTheta);
	dim3 blockLayout = dim3(nPhi);
	geometricPhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
		velPhi->getNextStepPitch() / sizeof(fReal), velTheta->getNextStepPitch() / sizeof(fReal),
		gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	velPhi->swapGPUBuffer();

	gridLayout = dim3(nTheta - 1);
	blockLayout = dim3(nPhi);
	geometricThetaKernel<<<gridLayout, blockLayout>>>
		(velTheta->getGPUNextStep(), velPhi->getGPUThisStep(), velTheta->getGPUThisStep(),
			velPhi->getNextStepPitch() / sizeof(fReal), velTheta->getNextStepPitch() / sizeof(fReal),
			gridLen, radius, timeStep);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	velTheta->swapGPUBuffer();
}