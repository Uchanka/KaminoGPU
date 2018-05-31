# include "KaminoHeader.cuh"

__global__ void crKernel(fReal *d_a, fReal *d_b, fReal *d_c, fReal *d_d, fReal *d_x)
{
	int thid = threadIdx.x;
	int blid = blockIdx.x;

	int stride = 1;

	int numThreads = blockDim.x;
	const unsigned int systemSize = blockDim.x * 2;

	int iteration = (int)log2(fReal(systemSize / 2));
#ifdef GPU_PRINTF 
	if (thid == 0 && blid == 0) printf("iteration = %d\n", iteration);
#endif

	__syncthreads();

	extern __shared__ char shared[];

	fReal* a = (fReal*)shared;
	fReal* b = (fReal*)&a[systemSize];
	fReal* c = (fReal*)&b[systemSize];
	fReal* d = (fReal*)&c[systemSize];
	fReal* x = (fReal*)&d[systemSize];

	a[thid] = d_a[thid + blid * systemSize];
	a[thid + blockDim.x] = d_a[thid + blockDim.x + blid * systemSize];

	b[thid] = d_b[thid + blid * systemSize];
	b[thid + blockDim.x] = d_b[thid + blockDim.x + blid * systemSize];

	c[thid] = d_c[thid + blid * systemSize];
	c[thid + blockDim.x] = d_c[thid + blockDim.x + blid * systemSize];

	d[thid] = d_d[thid + blid * systemSize];
	d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * systemSize];

	__syncthreads();

	//forward elimination
	for (int j = 0; j < iteration; j++)
	{
		__syncthreads();
		stride *= 2;
		int delta = stride / 2;

		if (threadIdx.x < numThreads)
		{
			int i = stride * threadIdx.x + stride - 1;
			int iLeft = i - delta;
			int iRight = i + delta;
			if (iRight >= systemSize) iRight = systemSize - 1;
			fReal tmp1 = a[i] / b[iLeft];
			fReal tmp2 = c[i] / b[iRight];
			b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
			d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
			a[i] = -a[iLeft] * tmp1;
			c[i] = -c[iRight] * tmp2;
		}
		numThreads /= 2;
	}

	if (thid < 2)
	{
		int addr1 = stride - 1;
		int addr2 = 2 * stride - 1;
		fReal tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
	}

	// backward substitution
	numThreads = 2;
	for (int j = 0; j < iteration; j++)
	{
		int delta = stride / 2;
		__syncthreads();
		if (thid < numThreads)
		{
			int i = stride * thid + stride / 2 - 1;
			if (i == delta - 1)
				x[i] = (d[i] - c[i] * x[i + delta]) / b[i];
			else
				x[i] = (d[i] - a[i] * x[i - delta] - c[i] * x[i + delta]) / b[i];
		}
		stride /= 2;
		numThreads *= 2;
	}

	__syncthreads();

	d_x[thid + blid * systemSize] = x[thid];
	d_x[thid + blockDim.x + blid * systemSize] = x[thid + blockDim.x];
}