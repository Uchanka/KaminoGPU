# include "../include/KaminoTimer.cuh"

KaminoTimer::KaminoTimer() : timeElapsed(0.0f)
{
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
}
KaminoTimer::~KaminoTimer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
void KaminoTimer::startTimer()
{
	cudaEventRecord(start, 0);
}
float KaminoTimer::stopTimer()
{
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	checkCudaErrors(cudaEventElapsedTime(&timeElapsed, start, stop));
	return timeElapsed;
}