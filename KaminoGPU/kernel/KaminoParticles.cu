# include "../include/KaminoParticles.cuh"

KaminoParticles::KaminoParticles(std::string path, size_t particlePGrid, size_t nTheta)
	: particlePGrid(particlePGrid), nPhi(2 * nTheta), nTheta(nTheta)
{
	if (path == "")
	{
		return;
	}
	cv::Mat image_In;
	image_In = cv::imread(path, cv::IMREAD_COLOR);
	if (!image_In.data)
	{
		std::cerr << "No particle image provided." << std::endl;
		return;
	}

	cv::Mat image_Flipped;
	cv::flip(image_In, image_Flipped, 1);

	cv::Mat image_Resized;
	cv::Size size(nPhi, nTheta);
	cv::resize(image_Flipped, image_Resized, size);

	numOfParticles = nPhi * nTheta * particlePGrid;
	coordCPUBuffer = new fReal[numOfParticles * 2];
	colorBGR = new fReal[numOfParticles * 3];

	checkCudaErrors(cudaMalloc(&coordGPUThisStep, sizeof(fReal) * numOfParticles * 2));
	checkCudaErrors(cudaMalloc(&coordGPUNextStep, sizeof(fReal) * numOfParticles * 2));

	for (size_t phi = 0; phi < nPhi; ++phi)
	{
		for (size_t theta = 0; theta < nTheta; ++theta)
		{
			cv::Point3_<uchar>* p = image_Resized.ptr<cv::Point3_<uchar>>(theta, phi);
			fReal B = p->x / 255.0; // B
			fReal G = p->y / 255.0; // G
			fReal R = p->z / 255.0; // R
			for (size_t part = 0; part < particlePGrid; ++part)
			{
				fReal phiCenter = static_cast<fReal>(phi) + 0.5 * (1.0 + std::sinf(rand() / 1000.0));
				fReal thetaCenter = static_cast<fReal>(theta) + 0.5 * (1.0 + std::cosf(rand() / 1000.0));
				fReal phiCoord = phiCenter * M_2PI / nPhi;
				fReal thetaCoord = thetaCenter * M_PI / nTheta;

				coordCPUBuffer[2 * (theta * nPhi + phi)] = phiCoord;
				coordCPUBuffer[2 * (theta * nPhi + phi) + 1] = thetaCoord;

				colorBGR[3 * (theta * nPhi + phi)] = B;
				colorBGR[3 * (theta * nPhi + phi) + 1] = G;
				colorBGR[3 * (theta * nPhi + phi) + 2] = R;
			}
		}
	}

	copy2GPU();
}

void KaminoParticles::copy2GPU()
{
	checkCudaErrors(cudaMemcpy(this->coordGPUThisStep, this->coordCPUBuffer,
		sizeof(fReal) * numOfParticles, cudaMemcpyHostToDevice));
}

void KaminoParticles::copyBack2CPU()
{
	checkCudaErrors(cudaMemcpy(this->coordCPUBuffer, this->coordGPUThisStep,
		sizeof(fReal) * numOfParticles, cudaMemcpyDeviceToHost));
}

void KaminoParticles::swapGPUBuffers()
{
	fReal* temp = this->coordGPUThisStep;
	this->coordGPUThisStep = this->coordGPUNextStep;
	this->coordGPUNextStep = temp;
}