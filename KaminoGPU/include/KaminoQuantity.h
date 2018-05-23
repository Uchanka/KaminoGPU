# pragma once

# include "KaminoHeader.h"

class KaminoQuantity
{
private:
	/* Name of the attribute */
	std::string attrName;

	/* Grid dimensions */
	size_t nPhi;
	size_t nTheta;

	/* Grid size */
	fReal gridLen;
	/* 1.0 / gridlen */
	fReal invGridLen;

	/* Staggered? */
	fReal phiOffset;
	fReal thetaOffset;

	/* Initial buffer at client side */
	fReal* cpuBuffer;
	/* Double buffer at server side */
	cudaArray* gpuThisStep;
	cudaArray* gpuNextStep;

	/* Get index */
	//size_t getIndex(size_t phi, size_t theta);

public:
	/* Constructor */
	KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
		fReal gridLen, fReal phiOffset, fReal thetaOffset);
	/* Destructor */
	~KaminoQuantity();

	/* Swap the GPU buffer */
	void swapGPUBuffer();
	/* Convert to texture */
	void createTexture();
	/* Copy the CPU end part to GPU */
	void copyToGPU();

	/* Get its name */
	std::string getName();
	/* Get phi dimension size */
	size_t getNPhi();
	/* Get theta dimension size */
	size_t getNTheta();
	/* Get the current step */
	fReal getCPUValueAt(size_t x, size_t y);
	/* Set the current step */
	void setCPUValueAt(size_t x, size_t y, fReal val);
	/* Access */
	fReal& accessCPUValueAt(size_t x, size_t y);
	/* Get the offset */
	fReal getPhiOffset();
	fReal getThetaOffset();

	static cudaChannelFormatDesc channelFormat;
};
