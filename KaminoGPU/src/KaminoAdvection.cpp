# include "../include/KaminoQuantity.h"


void KaminoSolver::advectAttrAt(KaminoQuantity* attr, size_t gridPhi, size_t gridTheta)
{
	KaminoQuantity* uPhi = (*this)["u"];
	KaminoQuantity* uTheta = (*this)["v"];

	fReal gPhi = attr->getPhiCoordAtIndex(gridPhi);
	fReal gTheta = attr->getThetaCoordAtIndex(gridTheta);

	fReal guPhi = uPhi->sampleAt(gPhi, gTheta, this->uNorthP, this->uSouthP);
	fReal guTheta = uTheta->sampleAt(gPhi, gTheta, this->uNorthP, this->uSouthP);

	fReal latRadius = this->radius * std::sin(gTheta);
	fReal cofPhi = timeStep / latRadius;
	fReal cofTheta = timeStep / radius;

	fReal deltaPhi = guPhi * cofPhi;
	fReal deltaTheta = guTheta * cofTheta;

	fReal midPhi = gPhi - 0.5 * deltaPhi;
	fReal midTheta = gTheta - 0.5 * deltaTheta;

	fReal muPhi = uPhi->sampleAt(midPhi, midTheta, this->uNorthP, this->uSouthP);
	fReal muTheta = uTheta->sampleAt(midPhi, midTheta, this->uNorthP, this->uSouthP);

	fReal averuPhi = 0.5 * (muPhi + guPhi);
	fReal averuTheta = 0.5 * (muTheta + guTheta);

	deltaPhi = averuPhi * cofPhi;
	deltaTheta = averuTheta * cofTheta;

	fReal pPhi = gPhi - deltaPhi;
	fReal pTheta = gTheta - deltaTheta;

	fReal advectedVal = attr->sampleAt(pPhi, pTheta, this->uNorthP, this->uSouthP);
	
	attr->writeValueTo(gridPhi, gridTheta, advectedVal);
}

void KaminoSolver::advectionScalar()
{
	for (auto quantity : this->centeredAttr)
	{
		KaminoQuantity* cenAttr = quantity.second;
# ifdef OMParallelize
# pragma omp parallel for
# endif
		for (int gridTheta = 0; gridTheta < cenAttr->getNTheta(); ++gridTheta)
		{
			for (size_t gridPhi = 0; gridPhi < cenAttr->getNPhi(); ++gridPhi)
			{
				advectAttrAt(cenAttr, gridPhi, gridTheta);
			}
		}
	}
}

void KaminoSolver::solvePolarVelocities()
{
	KaminoQuantity* uPhi = (*this)["u"];
	KaminoQuantity* uTheta = (*this)["v"];

	// First we derive velocity at the poles...
	size_t northernBelt = 0;
	size_t southernBelt = uPhi->getNTheta() - 1; // uTheta->getNTheta() - 2
	size_t northernPinch = 0;
	size_t southernPinch = uTheta->getNTheta() - 1;
	resetPoleVelocities();
	for (size_t gridPhi = 0; gridPhi < nPhi; ++gridPhi)
	{
		fReal phi = (M_2PI / nPhi) * gridPhi;

		size_t gridPhiP1 = (gridPhi + 1) % nPhi;
		fReal ootBeltUPhi = KaminoLerp(uPhi->getValueAt(gridPhi, northernBelt), uPhi->getValueAt(gridPhiP1, northernBelt), 0.5);
		fReal totBeltUPhi = KaminoLerp(uPhi->getValueAt(gridPhi, northernBelt + 1), uPhi->getValueAt(gridPhiP1, northernBelt + 1), 0.5);
		fReal uPhiLatLine = KaminoLerp(ootBeltUPhi, totBeltUPhi, 0.5);
		fReal uThetaLatLine = uTheta->getValueAt(gridPhi, northernPinch + 1);

		uNorthP[x] += uThetaLatLine * std::cos(phi) - uPhiLatLine * std::sin(phi);
		uNorthP[y] += uThetaLatLine * std::sin(phi) + uPhiLatLine * std::cos(phi);


		ootBeltUPhi = KaminoLerp(uPhi->getValueAt(gridPhi, southernBelt), uPhi->getValueAt(gridPhiP1, southernBelt), 0.5);
		totBeltUPhi = KaminoLerp(uPhi->getValueAt(gridPhi, southernBelt - 1), uPhi->getValueAt(gridPhiP1, southernBelt - 1), 0.5);
		uPhiLatLine = KaminoLerp(ootBeltUPhi, totBeltUPhi, 0.5);
		uThetaLatLine = uTheta->getValueAt(gridPhi, southernPinch - 1);


		uSouthP[x] += -uThetaLatLine * std::cos(phi) - uPhiLatLine * std::sin(phi);
		uSouthP[y] += -uThetaLatLine * std::sin(phi) + uPhiLatLine * std::cos(phi);
	}
	averageVelocities();
	//Now we have the projected x, y components at polars
	for (size_t gridPhi = 0; gridPhi < nPhi; ++gridPhi)
	{
		fReal phi = (M_2PI / nPhi) * gridPhi;
		fReal northernUTheta = uNorthP[x] * std::cos(phi) + uNorthP[y] * std::sin(phi);
		uTheta->writeValueTo(gridPhi, northernPinch, northernUTheta);
		fReal southernUTheta = -uSouthP[x] * std::cos(phi) - uSouthP[y] * std::sin(phi);
		uTheta->writeValueTo(gridPhi, southernPinch, southernUTheta);
	}
}

void KaminoSolver::advectionSpeed()
{
	//Advect as is for uPhi
	KaminoQuantity* uPhi = (*this)["u"];
# ifdef OMParallelize
# pragma omp parallel for
# endif
	for (int gridTheta = 0; gridTheta < uPhi->getNTheta(); ++gridTheta)
	{
		for (size_t gridPhi = 0; gridPhi < uPhi->getNPhi(); ++gridPhi)
		{
			advectAttrAt(uPhi, gridPhi, gridTheta);
		}
	}

	//Tread carefully for uTheta...
	KaminoQuantity* uTheta = (*this)["v"];
	// Apart from the poles...
# ifdef OMParallelize
# pragma omp parallel for
# endif
	for (int gridTheta = 1; gridTheta < uTheta->getNTheta() - 1; ++gridTheta)
	{
		for (size_t gridPhi = 0; gridPhi < uTheta->getNPhi(); ++gridPhi)
		{
			advectAttrAt(uTheta, gridPhi, gridTheta);
		}
	}
	/// TODO
	solvePolarVelocities();
}

void KaminoSolver::resetPoleVelocities()
{
	for (unsigned i = 0; i < 2; ++i)
	{
		uNorthP[i] = 0.0;
		uSouthP[i] = 0.0;
	}
}

void KaminoSolver::averageVelocities()
{
	for (unsigned i = 0; i < 2; ++i)
	{
		uNorthP[i] /= this->nPhi;
		uSouthP[i] /= this->nPhi;
	}
}
