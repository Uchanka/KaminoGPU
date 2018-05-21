# include "../include/KaminoQuantity.h"
# include "../include/CubicSolver.h"

// CONSTRUCTOR / DESTRUCTOR >>>>>>>>>>

KaminoSolver::KaminoSolver(size_t nPhi, size_t nTheta, fReal radius, fReal gridLength, fReal frameDuration,
	fReal A, int B, int C, int D, int E) :
	nPhi(nPhi), nTheta(nTheta), radius(radius), gridLen(gridLength), invGridLen(1.0 / gridLength), frameDuration(frameDuration),
	timeStep(0.0), timeElapsed(0.0),
	A(A), B(B), C(C), D(D), E(E)
{
	this->beffourierF = new fReal[nPhi * nTheta];
	this->fourieredFReal = new fReal[nPhi * nTheta];
	this->fourieredFImag = new fReal[nPhi * nTheta];
	this->fourierUReal = new fReal[nPhi * nTheta];
	this->fourierUImag = new fReal[nPhi * nTheta];

	this->a = new fReal[nTheta];
	this->b = new fReal[nTheta];
	this->c = new fReal[nTheta];
	this->dReal = new fReal[nTheta];
	this->dImag = new fReal[nTheta];

	addStaggeredAttr("u", -0.5, 0.5);		// u velocity
	addStaggeredAttr("v", 0.0, 0.0);		// v velocity
	addCenteredAttr("p", 0.0, 0.5);			// p pressure
	addCenteredAttr("density", 0.0, 0.5);	// density

	this->gridTypes = new gridType[nPhi * nTheta];
	
	initialize_velocity();
	initialize_pressure();
	initialize_density();
	//initialize_boundary();
}

KaminoSolver::~KaminoSolver()
{
	delete[] beffourierF;
	delete[] fourieredFReal;
	delete[] fourieredFImag;
	delete[] fourierUReal;
	delete[] fourierUImag;

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] dReal;
	delete[] dImag;

	for (auto& attr : this->centeredAttr)
	{
		delete attr.second;
	}
	for (auto& attr : this->staggeredAttr)
	{
		delete attr.second;
	}
	delete[] this->gridTypes;
}

void KaminoSolver::stepForward(fReal timeStep)
{
	this->timeStep = timeStep;
	advectionScalar();
	advectionSpeed();
	//std::cout << "Advection completed" << std::endl;
	this->swapAttrBuffers();

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
	fReal gravity = 9.8;
	KaminoQuantity* v = staggeredAttr["v"];

	for(size_t j = 0; j < nTheta + 1; ++j){
		for(size_t i = 0; i < nPhi; ++i){
			fReal vBeforeUpdate = v->getValueAt(i, j);
			fReal theta = j*gridLen;
			v->writeValueTo(i, j, vBeforeUpdate + gravity * sin(theta) * timeStep);
		}
	}

	v->swapBuffer();
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

/* Duplicate of getIndex() in KaminoQuantity */
size_t KaminoSolver::getIndex(size_t x, size_t y)
{
	return y * nPhi + x;
}

gridType KaminoSolver::getGridTypeAt(size_t x, size_t y)
{
	return gridTypes[getIndex(x, y)];
}


void KaminoSolver::addCenteredAttr(std::string name, fReal xOffset, fReal yOffset)
{
	size_t attrnPhi = this->nPhi;
	size_t attrnTheta = this->nTheta;

	KaminoQuantity* ptr = new KaminoQuantity(name, attrnPhi, attrnTheta, this->gridLen, xOffset, yOffset);
	this->centeredAttr.emplace(std::pair<std::string, KaminoQuantity*>(name, ptr));
}

void KaminoSolver::addStaggeredAttr(std::string name, fReal xOffset, fReal yOffset)
{
	size_t attrnPhi = this->nPhi;
	size_t attrnTheta = this->nTheta;
	// Is the staggered attribute uTheta?
	if (name == "v")
	{
		attrnTheta += 1;
	}
	KaminoQuantity* ptr = new KaminoQuantity(name, attrnPhi, attrnTheta, this->gridLen, xOffset, yOffset);
	this->staggeredAttr.emplace(std::pair<std::string, KaminoQuantity*>(name, ptr));
}

KaminoQuantity* KaminoSolver::getAttributeNamed(std::string name)
{
	return (*this)[name];
}

KaminoQuantity* KaminoSolver::operator[](std::string name)
{
	if (centeredAttr.find(name) == centeredAttr.end())
	{
		return staggeredAttr.at(name);
	}
	else
	{
		return centeredAttr.at(name);
	}
}

void KaminoSolver::swapAttrBuffers()
{
	for (auto quantity : this->centeredAttr)
	{
		quantity.second->swapBuffer();
	}
	for (auto quantity : this->staggeredAttr)
	{
		quantity.second->swapBuffer();
	}
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