# include "../include/KaminoQuantity.h"

void KaminoSolver::initialize_pressure()
{
	for (size_t i = 0; i < nPhi; ++i) {
		for (size_t j = 0; j < nTheta; ++j) {
			centeredAttr["p"]->setValueAt(i, j, 0.0);
		}
	}
}

void KaminoSolver::initialize_velocity()
{
	KaminoQuantity* u = this->staggeredAttr["u"];
	KaminoQuantity* v = this->staggeredAttr["v"];

	fReal f = 0.0;
	fReal g = 0.0;
	size_t sizePhi = u->getNPhi();
	size_t sizeTheta = u->getNTheta();
	for (size_t j = 0; j < sizeTheta; ++j) {
		for (size_t i = 0; i < sizePhi; ++i) {
			f = fPhi(i * gridLen);
			g = gTheta(j * gridLen);
			u->setValueAt(i, j, A + (f * g));
		}
	}

	fReal l = 0.0;
	fReal m = 0.0;
	sizePhi = v->getNPhi();
	sizeTheta = v->getNTheta();
	
	// rest of sphere
	for (size_t j = 1; j < sizeTheta - 1; ++j) {
		for (size_t i = 0; i < sizePhi; ++i) {
			l = lPhi(i * gridLen);
			m = mTheta(j * gridLen);
			v->setValueAt(i, j, (l * m));
		}
	}

	// Heat up the next buffer.
	for (size_t j = 0; j < u->getNTheta(); ++j) {
		for (size_t i = 0; i < u->getNPhi(); ++i) {
			u->writeValueTo(i, j, u->getValueAt(i, j));
		}
	}
	for (size_t j = 1; j < v->getNTheta() - 1; ++j) {
		for (size_t i = 0; i < v->getNPhi(); ++i) {
			v->writeValueTo(i, j, v->getValueAt(i, j));
		}
	}

	solvePolarVelocities();
	u->swapBuffer();
	v->swapBuffer();
}


fReal KaminoSolver::fPhi(const fReal x)
{
	fReal arg = x;
	return sin(arg) + 0.1 * B * std::cos(std::rand());
}

fReal KaminoSolver::gTheta(const fReal y)
{
	fReal arg = y;
	return cos(arg) + 0.1 * C * std::sin(std::rand());
}

fReal KaminoSolver::lPhi(const fReal x)
{
	fReal arg = x;
	return cos(arg) + 0.1 * D * std::sin(std::rand());
}

fReal KaminoSolver::mTheta(const fReal y)
{
	fReal arg = y;
	return sin(arg) + 0.1 * E * std::cos(std::rand());
}

fReal KaminoSolver::FBM(const fReal x, const fReal y) {
	fReal total = 0.0f;
	fReal resolution = 1.0;
	fReal persistance = 0.5;
	int octaves = 4;

	for (int i = 0; i < octaves; i++) {
		fReal freq = std::pow(2.0f, i);
		fReal amp = std::pow(persistance, i);
		total += amp * interpNoise2D(x * freq / resolution, y * freq / resolution);
	}
	fReal a = 1 - persistance;  // normalization

	return a * total / 2.0f;  // normalized, pseudorandom number between -1 and 1
}

fReal KaminoSolver::interpNoise2D(const fReal x, const fReal y) const {
	fReal intX = std::floor(x);
	fReal fractX = x - intX;
	fReal intY = std::floor(y);
	fReal fractY = y - intY;

	fReal v1 = rand(Eigen::Matrix<fReal, 2, 1>(intX, intY));
	fReal v2 = rand(Eigen::Matrix<fReal, 2, 1>(intX + 1, intY));
	fReal v3 = rand(Eigen::Matrix<fReal, 2, 1>(intX, intY + 1));
	fReal v4 = rand(Eigen::Matrix<fReal, 2, 1>(intX + 1, intY + 1));

	// interpolate for smooth transitions
	fReal i1 = KaminoLerp(v1, v2, fractX);
	fReal i2 = KaminoLerp(v3, v4, fractX);
	return KaminoLerp(i1, i2, fractY);
}

fReal KaminoSolver::rand(const Eigen::Matrix<fReal, 2, 1> vecA) const {
	// return pseudorandom number between -1 and 1
	Eigen::Matrix<fReal, 2, 1> vecB = Eigen::Matrix<fReal, 2, 1>(12.9898, 4.1414);
	fReal val = sin(vecA.dot(vecB) * 43758.5453);
	return val - std::floor(val);
}

void KaminoSolver::initialize_density()
{
	for(size_t i = 0; i < nPhi; ++i)
	{
		for(size_t j = 0; j < nTheta; ++j)
		{
			centeredAttr["density"]->setValueAt(i, j, 0.0);
		}
	}
}

void KaminoSolver::initialize_boundary()
{
	for (size_t gridX = 0; gridX != this->nPhi / 2; ++gridX)
	{
		//this->gridTypes[getIndex(gridX, nTheta / 2)] = SOLIDGRID;
		//this->gridTypes[getIndex(gridX, this->nTheta - 1)] = SOLIDGRID;
	}
	/*for (size_t gridY = 0; gridY != this->nTheta; ++gridY)
	{
		this->gridTypes[getIndex(0, gridY)] = SOLIDGRID;
		this->gridTypes[getIndex(nPhi / 2, gridY)] = SOLIDGRID;
	}*/
}
