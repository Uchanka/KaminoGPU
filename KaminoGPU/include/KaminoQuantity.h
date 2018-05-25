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
	/* Double pitch buffer at server side */
	fReal* gpuThisStep;
	size_t thisStepPitch;
	fReal* gpuNextStep;
	size_t nextStepPitch;

	cudaChannelFormatDesc desc;
	/* Get index */
	//size_t getIndex(size_t phi, size_t theta);

public:
	/* Constructor */
	KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
		fReal phiOffset, fReal thetaOffset);
	/* Destructor */
	~KaminoQuantity();

	/* Swap the GPU buffer */
	void swapGPUBuffer();
	/* Copy the CPU end part to GPU */
	void copyToGPU();
	/* Copy backwards */
	void copyBackToCPU();
	/* Bind the texture */
	void bindTexture(table2D& tex);
	void unbindTexture(table2D& tex);

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
	fReal* getGPUThisStep();
	fReal* getGPUNextStep();

	size_t getThisStepPitch();
	size_t getNextStepPitch();

	static cudaChannelFormatDesc channelFormat;
};
