# include "../include/KaminoSolver.cuh"

static table2D texGeoVelPhi;
static table2D texGeoVelTheta;

static __constant__ size_t nPhiGlobal;
static __constant__ size_t nThetaGlobal;
static __constant__ fReal radiusGlobal;
static __constant__ fReal timeStepGlobal;
static __constant__ fReal gridLenGlobal;

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

__global__ void geometricPhiKernel
(fReal* velPhiOutput, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLenGlobal;
	// The factor
	fReal factor = timeStepGlobal * cosf(gTheta) / (radiusGlobal * sinf(gTheta));

	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhiGlobal;
	fReal gThetaTex = (fReal)thetaId / nThetaGlobal;

	// Sample the speed
	fReal guPhi = tex2D(texGeoVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D(texGeoVelTheta, gPhiTex, gThetaTex);

	fReal updateduPhi = guPhi - factor * guPhi * gTheta;

	velPhiOutput[thetaId * nPitchInElements + phiId] = updateduPhi;
};

__global__ void geometricThetaKernel
(fReal* velThetaOutput, size_t nPitchInElements)
{
	// Index
	int splitVal = nPhiGlobal / blockDim.x;
	int threadSequence = blockIdx.x % splitVal;
	int phiId = threadIdx.x + threadSequence * blockDim.x;
	int thetaId = blockIdx.x / splitVal;
	// Coord in phi-theta space
	fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLenGlobal;
	// The factor
	fReal factor = timeStepGlobal * cosf(gTheta) / (radiusGlobal * sinf(gTheta));

	// Coord in u-v texture space
	fReal gPhiTex = (fReal)phiId / nPhiGlobal;
	fReal gThetaTex = (fReal)thetaId / nThetaGlobal;

	// Sample the speed
	fReal guPhi = tex2D(texGeoVelPhi, gPhiTex, gThetaTex);
	fReal guTheta = tex2D(texGeoVelTheta, gPhiTex, gThetaTex);

	fReal updateduTheta = guTheta + factor * guPhi * guPhi;

	velThetaOutput[thetaId * nPitchInElements + phiId] = updateduTheta;
}

void KaminoSolver::geometric()
{
	setTextureParams(&texGeoVelPhi);
	setTextureParams(&texGeoVelTheta);
	velPhi->bindTexture(&texGeoVelPhi);
	velTheta->bindTexture(&texGeoVelTheta);



	checkCudaErrors(cudaMemcpyToSymbol(nPhiGlobal, (&this->nPhi), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(nThetaGlobal, (&this->nTheta), sizeof(size_t)));
	checkCudaErrors(cudaMemcpyToSymbol(radiusGlobal, (&this->radius), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(timeStepGlobal, (&this->timeStep), sizeof(fReal)));
	checkCudaErrors(cudaMemcpyToSymbol(gridLenGlobal, (&this->gridLen), sizeof(fReal)));



	dim3 gridLayout;
	dim3 blockLayout;
	determineLayout(gridLayout, blockLayout, nTheta - 1, nPhi);
	geometricPhiKernel<<<gridLayout, blockLayout>>>
	(velPhi->getGPUNextStep(), velPhi->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	geometricThetaKernel<<<gridLayout, blockLayout>>>
	(velTheta->getGPUNextStep(), velTheta->getNextStepPitchInElements());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	velPhi->unbindTexture(&texGeoVelPhi);
	velTheta->unbindTexture(&texGeoVelTheta);
	swapVelocityBuffers();
}