# include "../include/KaminoSolver.h"
# include "../include/CubicSolver.h"

// CONSTRUCTOR / DESTRUCTOR >>>>>>>>>>

KaminoSolver::KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal gridLength, fReal frameDuration,
	fReal A, int B, int C, int D, int E) :
	nPhi(nPhi), nTheta(nTheta), radius(radius), gridLen(gridLength), invGridLen(1.0 / gridLength), frameDuration(frameDuration),
	timeStep(0.0), timeElapsed(0.0),
	A(A), B(B), C(C), D(D), E(E)
{
	/// Replace it later with functions from helper_cuda.h!
	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors(cudaMalloc((void **)&gpuUPool,
		sizeof(Complex) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)&gpuFPool,
		sizeof(Complex) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)(&gpuA),
		sizeof(fReal) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)(&gpuB),
		sizeof(fReal) * nPhi * nTheta));
	checkCudaErrors(cudaMalloc((void **)(&gpuC),
		sizeof(fReal) * nPhi * nTheta));
	precomputeABCCoef();

	this->velPhi = new KaminoQuantity("velPhi", nPhi, nTheta,
		gridLen, -0.5, 0.5);
	this->velTheta = new KaminoQuantity("velTheta", nPhi, nTheta - 1,
		gridLen, 0.0, 1.0);

	this->cpuGridTypesBuffer = new gridType[nPhi * nTheta];
	checkCudaErrors(cudaMalloc((void **)(this->gpuGridTypes),
		sizeof(gridType) * nPhi * nTheta));
	
	initialize_velocity();
	copyVelocity2GPU();

	initialize_boundary();
	copyGridType2GPU();
}

KaminoSolver::~KaminoSolver()
{
	checkCudaErrors(cudaFree(gpuFPool));
	checkCudaErrors(cudaFree(gpuUPool));
	checkCudaErrors(cudaFree(gpuA));
	checkCudaErrors(cudaFree(gpuB));
	checkCudaErrors(cudaFree(gpuC));

	delete this->velPhi;
	delete this->velTheta;

	delete[] cpuGridTypesBuffer;
	checkCudaErrors(cudaFree(gpuGridTypes));
}

void KaminoSolver::copyVelocity2GPU()
{
	velPhi->copyToGPU();
	velTheta->copyToGPU();
}

void KaminoSolver::bindPressure2Tex()
{

}
void KaminoSolver::bindVelocity2Tex()
{

}
void KaminoSolver::defineTextureTable()
{
	texVelPhi.addressMode[0] = cudaAddressModeWrap;
	texVelPhi.addressMode[1] = cudaAddressModeMirror;
	texVelPhi.filterMode = cudaFilterModeLinear;
	texVelPhi.normalized = true;    // access with normalized texture coordinates

	texVelTheta.addressMode[0] = cudaAddressModeWrap;
	texVelTheta.addressMode[1] = cudaAddressModeMirror;
	texVelTheta.filterMode = cudaFilterModeLinear;
	texVelTheta.normalized = true;    // access with normalized texture coordinates

	texPressure.addressMode[0] = cudaAddressModeWrap;
	texPressure.addressMode[1] = cudaAddressModeMirror;
	texPressure.filterMode = cudaFilterModeLinear;
	texPressure.normalized = true;    // access with normalized texture coordinates
}

void KaminoSolver::precomputeABCCoef()
{
	fReal* cpuABuffer = new fReal[nTheta];
	fReal* cpuBBuffer = new fReal[nTheta];
	fReal* cpuCBuffer = new fReal[nTheta];

	for (size_t thetaI = 0; thetaI < nTheta; ++thetaI)
	{
	}
}

void KaminoSolver::stepForward(fReal timeStep)
{
	this->timeStep = timeStep;
	advection();
	//std::cout << "Advection completed" << std::endl;
	swapAttrBuffers();
	geometric(); // Buffer is swapped here
	//std::cout << "Geometric completed" << std::endl;
	//bodyForce();
	projection();
	//std::cout << "Projection completed" << std::endl;
	this->timeElapsed += timeStep;
}

// Phi: 0 - 2pi  Theta: 0 - pi
bool validatePhiTheta(fReal & phi, fReal & theta)
{
	int loops = static_cast<int>(std::floor(theta / M_2PI));
	theta = theta - loops * M_2PI;
	// Now theta is in 0-2pi range

	bool isFlipped = false;
	
	if (theta > M_PI)
	{
		theta = M_2PI - theta;
		phi += M_PI;
		isFlipped = true;
	}

	loops = static_cast<int>(std::floor(phi / M_2PI));
	phi = phi - loops * M_2PI;
	// Now phi is in 0-2pi range

	return isFlipped;
}

void KaminoSolver::bodyForce()
{
	/// This is just a place holder now...
}

/* Tri-diagonal matrix solver */
void KaminoSolver::TDMSolve(fReal* a, fReal* b, fReal* c, fReal* d)
{
	// |b0 c0 0 ||x0| |d0|
 	// |a1 b1 c1||x1|=|d1|
 	// |0  a2 b2||x2| |d2|

    int n = nTheta;
    n--; // since we index from 0
    c[0] /= b[0];
    d[0] /= b[0];

	for (int i = 1; i < n; i++) {
        c[i] /= b[i] - a[i]*c[i-1];
        d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]);
    }

    d[n] = (d[n] - a[n]*d[n-1]) / (b[n] - a[n]*c[n-1]);

    for (int i = n; i-- > 0;) {
        d[i] -= c[i]*d[i+1];
    }
}

gridType KaminoSolver::getGridTypeAt(size_t x, size_t y)
{
	return this->cpuGridTypesBuffer[getIndex(x, y)];
}

KaminoQuantity* KaminoSolver::getAttributeNamed(std::string name)
{
	return (*this)[name];
}

void KaminoSolver::swapAttrBuffers()
{
	
}


// <<<<<<<<<<
// OUTPUT >>>>>>>>>>


void KaminoSolver::write_data_bgeo(const std::string& s, const int frame)
{
	std::string file = s + std::to_string(frame) + ".bgeo";
	std::cout << "Writing to: " << file << std::endl;

	Partio::ParticlesDataMutable* parts = Partio::create();
	Partio::ParticleAttribute pH, vH, psH, dens;
	pH = parts->addAttribute("position", Partio::VECTOR, 3);
	vH = parts->addAttribute("v", Partio::VECTOR, 3);
	psH = parts->addAttribute("pressure", Partio::VECTOR, 1);
	dens = parts->addAttribute("density", Partio::VECTOR, 1);

	Eigen::Matrix<float, 3, 1> pos;
	Eigen::Matrix<float, 3, 1> vel;
	fReal pressure, densityValue;
	fReal velX, velY;

	KaminoQuantity* u = staggeredAttr["u"];
	KaminoQuantity* v = staggeredAttr["v"];
	fReal uRight, uLeft, vUp, vDown;

	size_t upi, vpi;

	for (size_t j = 0; j < nTheta; ++j) {
		for (size_t i = 0; i < nPhi; ++i) {
			uLeft = u->getValueAt(i, j);
			i == (nPhi - 1) ? upi = 0 : upi = i + 1;
			vDown = v->getValueAt(i, j);
			j == (nTheta - 1) ? vpi = 0 : vpi = j + 1;
			uRight = u->getValueAt(upi, j);
			vUp = u->getValueAt(i, vpi);

			velX = (uLeft + uRight) / 2.0;
			velY = (vUp + vDown) / 2.0;

			pos = Eigen::Matrix<float, 3, 1>(i * gridLen, j * gridLen, 0.0);
			vel = Eigen::Matrix<float, 3, 1>(0.0, velY, velX);
			mapVToSphere(pos, vel);
			mapPToSphere(pos);

			pressure = centeredAttr["p"]->getValueAt(i, j);
			densityValue = centeredAttr["density"]->getValueAt(i, j);
			
			int idx = parts->addParticle();
			float* p = parts->dataWrite<float>(pH, idx);
			float* v = parts->dataWrite<float>(vH, idx);
			float* ps = parts->dataWrite<float>(psH, idx);
			float* de = parts->dataWrite<float>(dens, idx);

			ps[0] = density * radius * pressure / timeStep;
			de[0] = densityValue;

			for (int k = 0; k < 3; ++k) {
				p[k] = pos(k, 0);
				v[k] = vel(k, 0);
			}
		}
	}

	Partio::write(file.c_str(), *parts);
	parts->release();
}

void KaminoSolver::mapPToSphere(Eigen::Matrix<float, 3, 1>& pos) const
{
	float theta = pos[1];
	float phi = pos[0];
	pos[0] = radius * sin(theta) * cos(phi);
	pos[2] = radius * sin(theta) * sin(phi);
	pos[1] = radius * cos(theta);
}

void KaminoSolver::mapVToSphere(Eigen::Matrix<float, 3, 1>& pos, Eigen::Matrix<float, 3, 1>& vel) const
{
	float theta = pos[1];
	float phi = pos[0];

	float u_theta = vel[1];
	float u_phi = vel[2];

	vel[0] = cos(theta) * cos(phi) * u_theta - sin(phi) * u_phi;
	vel[2] = cos(theta) * sin(phi) * u_theta + cos(phi) * u_phi;
	vel[1] = -sin(theta) * u_theta;
}

void KaminoSolver::mapToCylinder(Eigen::Matrix<float, 3, 1>& pos) const
{
	//float radius = 5.0;
	float phi = 2*M_PI*pos[0] / (nPhi * gridLen);
	float z = pos[1];
	pos[0] = radius * cos(phi);
	pos[1] = radius * sin(phi);
	pos[2] = z;
}

gridType* KaminoSolver::getGridTypeHandle()
{
	return this->gridTypes;
}