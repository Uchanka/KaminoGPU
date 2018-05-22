# include "../include/KaminoQuantity.h"

KaminoQuantity::KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta, 
	fReal gridLen, fReal phiOffset, fReal thetaOffset)
	: nPhi(nPhi), nTheta(nTheta), gridLen(gridLen), invGridLen(1.0 / gridLen), 
	attrName(attributeName), phiOffset(phiOffset), thetaOffset(thetaOffset)
{
	cpuBuffer = new fReal[nPhi * nTheta];

	checkCudaErrors(cudaMalloc((void**)&gpuThisStep, nPhi * nTheta * sizeof(fReal)));
	checkCudaErrors(cudaMalloc((void**)&gpuNextStep, nPhi * nTheta * sizeof(fReal)));
}

KaminoQuantity::~KaminoQuantity()
{
	delete[] cpuBuffer;

	checkCudaErrors(cudaFree(gpuThisStep));
	checkCudaErrors(cudaFree(gpuNextStep));
}

std::string KaminoQuantity::getName()
{
	return this->attrName;
}

size_t KaminoQuantity::getNPhi()
{
	return this->nPhi;
}

size_t KaminoQuantity::getNTheta()
{
	return this->nTheta;
}

void KaminoQuantity::swapGPUBuffer()
{
	fReal* tempPtr = this->gpuThisStep;
	this->gpuThisStep = this->gpuNextStep;
	this->gpuNextStep = tempPtr;
}

fReal KaminoQuantity::getCPUValueAt(size_t phi, size_t theta)
{
	return this->accessCPUValueAt(phi, theta);
}

void KaminoQuantity::setCPUValueAt(size_t phi, size_t theta, fReal val)
{
	this->accessCPUValueAt(phi, theta) = val;
}

fReal& KaminoQuantity::accessCPUValueAt(size_t phi, size_t theta)
{
	return this->cpuBuffer[getIndex(phi, theta)];
}

fReal KaminoQuantity::getThetaOffset()
{
	return this->thetaOffset;
}

fReal KaminoQuantity::getPhiOffset()
{
	return this->phiOffset;
}

void KaminoQuantity::copyToGPU()
{
	checkCudaErrors(cudaMemcpy((void*)this->cpuBuffer, (void*)this->gpuThisStep, 
		sizeof(fReal) * nTheta * nPhi, ::cudaMemcpyHostToDevice));
}