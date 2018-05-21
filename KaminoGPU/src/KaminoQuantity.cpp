# include "../include/KaminoQuantity.h"

KaminoQuantity::KaminoQuantity(std::string attributeName, size_t nPhi, size_t nTheta, fReal gridLen, fReal xOffset, fReal yOffset)
	: nPhi(nPhi), nTheta(nTheta), gridLen(gridLen), invGridLen(1.0 / gridLen), attrName(attributeName), xOffset(xOffset), yOffset(yOffset)
{
	thisStep = new fReal[nPhi * nTheta];
	nextStep = new fReal[nPhi * nTheta];
}

KaminoQuantity::~KaminoQuantity()
{
	delete[] thisStep;
	delete[] nextStep;
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

void KaminoQuantity::swapBuffer()
{
	fReal* tempPtr = this->thisStep;
	this->thisStep = this->nextStep;
	this->nextStep = tempPtr;
}

fReal KaminoQuantity::getValueAt(size_t x, size_t y)
{
	return this->accessValueAt(x, y);
}

void KaminoQuantity::setValueAt(size_t x, size_t y, fReal val)
{
	this->accessValueAt(x, y) = val;
}

fReal& KaminoQuantity::accessValueAt(size_t x, size_t y)
{
	return this->thisStep[getIndex(x, y)];
}

void KaminoQuantity::writeValueTo(size_t x, size_t y, fReal val)
{
/*# ifdef DEBUGBUILD
	if (val > 1e4)
	{
		std::cerr << "Explosion detected " << std::endl;
	}
# endif*/
	this->nextStep[getIndex(x, y)] = val;
}

size_t KaminoQuantity::getIndex(size_t x, size_t y)
{
# ifdef DEBUGBUILD
	if (x >= this->nPhi || y >= this->nTheta)
	{
		std::cerr << "Index out of bound at x: " << x << " y: " << y << std::endl;
	}
# endif
	return y * nPhi + x;
}

fReal KaminoQuantity::getPhiCoordAtIndex(size_t x)
{
	fReal xFloat = static_cast<fReal>(x) + xOffset;
	return xFloat * this->gridLen;
}
// Might be problematic : what if phi - xOffset < 0.0 due to a float point error?
size_t KaminoQuantity::getPhiIndexAtCoord(fReal phi)
{
	fReal phiInt = phi * this->invGridLen;
	return static_cast<size_t>(phiInt - xOffset);
}

fReal KaminoQuantity::getThetaCoordAtIndex(size_t y)
{
	fReal yFloat = static_cast<fReal>(y) + yOffset;
	return yFloat * this->gridLen;
}
// Might be problematic as well : what if phi - xOffset < 0.0 due to a float point error?
size_t KaminoQuantity::getThetaIndexAtCoord(fReal theta)
{
	fReal thetaInt = theta * this->invGridLen;
	return static_cast<size_t>(thetaInt - yOffset);
}

/*
Bilinear interpolated for now.
*/
fReal KaminoQuantity::sampleAt(fReal x, fReal y, fReal uNorthP[2], fReal uSouthP[2])
{
	fReal phi = x - gridLen * this->xOffset;
	fReal theta = y - gridLen * this->yOffset;
	// Phi and Theta are now shifted back to origin

	bool isFlippedPole = validatePhiTheta(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(std::floor(normedPhi));
	int thetaIndex = static_cast<int>(std::floor(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if (thetaIndex == 0 && isFlippedPole) // If it's not flipped the theta+1 belt would just be belt 1
	{
		if (attrName == "u")
		{
			alphaTheta = 2.0 * alphaTheta;
			size_t phiLower = (phiIndex + nPhi / 2) % nPhi;
			size_t phiHigher = (phiLower + 1) % nPhi;
			fReal lowerBelt = KaminoLerp(getValueAt(phiLower, 0), getValueAt(phiHigher, 0), alphaPhi);

			fReal lowerPhi = (phiLower - 0.5) * gridLen;
			fReal higherPhi = (phiLower + 0.5) * gridLen;
			fReal loweruPhi = -uNorthP[0] * std::cos(lowerPhi) + uNorthP[1] * std::sin(lowerPhi);
			fReal higheruPhi = -uNorthP[0] * std::cos(higherPhi) + uNorthP[1] * std::sin(higherPhi);
			fReal higherBelt = KaminoLerp(loweruPhi, higheruPhi, alphaPhi);

			fReal lerped = KaminoLerp(lowerBelt, higherBelt, alphaTheta);
			return lerped;
		}
		else
		{
			//Lower is to the opposite, higher is on this side
			alphaTheta = 1.0 - alphaTheta;
			size_t phiLower = phiIndex % nPhi;
			size_t phiHigher = (phiLower + 1) % nPhi;
			size_t phiLowerOppo = (phiLower + nPhi / 2) % nPhi;
			size_t phiHigherOppo = (phiHigher + nPhi / 2) % nPhi;

			fReal lowerBelt = KaminoLerp<fReal>(getValueAt(phiLower, 0), getValueAt(phiHigher, 0), alphaPhi);
			fReal higherBelt = KaminoLerp<fReal>(getValueAt(phiLowerOppo, 0), getValueAt(phiHigherOppo, 0), alphaPhi);
			fReal lerped = KaminoLerp<fReal>(lowerBelt, higherBelt, alphaTheta);
			return lerped;
		}
	}
	else if (thetaIndex == nTheta - 1)
	{
		if (attrName == "u")
		{
			alphaTheta = 2.0 * alphaTheta;
			size_t phiLower = phiIndex % nPhi;
			size_t phiHigher = (phiLower + 1) % nPhi;
			fReal lowerBelt = KaminoLerp(getValueAt(phiLower, thetaIndex), getValueAt(phiHigher, thetaIndex), alphaPhi);
			
			fReal lowerPhi = (phiLower - 0.5) * gridLen;
			fReal higherPhi = (phiLower + 0.5) * gridLen;
			fReal loweruPhi = -uSouthP[0] * std::cos(lowerPhi) + uSouthP[1] * std::sin(lowerPhi);
			fReal higheruPhi = -uSouthP[0] * std::cos(higherPhi) + uSouthP[1] * std::sin(higherPhi);
			fReal higherBelt = KaminoLerp(loweruPhi, higheruPhi, alphaPhi);

			fReal lerped = KaminoLerp(lowerBelt, higherBelt, alphaTheta);
			return lerped;
		}
		else
		{
			//Lower is on this side, higher is to the opposite
			size_t phiLower = phiIndex % nPhi;
			size_t phiHigher = (phiLower + 1) % nPhi;
			size_t phiLowerOppo = (phiLower + nPhi / 2) % nPhi;
			size_t phiHigherOppo = (phiHigher + nPhi / 2) % nPhi;

			fReal lowerBelt = KaminoLerp<fReal>(getValueAt(phiLower, nTheta - 1), getValueAt(phiHigher, nTheta - 1), alphaPhi);
			fReal higherBelt = KaminoLerp<fReal>(getValueAt(phiLowerOppo, nTheta - 1), getValueAt(phiHigherOppo, nTheta - 1), alphaTheta);

			fReal lerped = KaminoLerp<fReal>(lowerBelt, higherBelt, alphaTheta);
			return lerped;
		}
	}
	else
	{
		size_t phiLower = phiIndex % nPhi;
		size_t phiHigher = (phiLower + 1) % nPhi;
		size_t thetaLower = thetaIndex;
		size_t thetaHigher = thetaIndex + 1;

		fReal lowerBelt = KaminoLerp<fReal>(getValueAt(phiLower, thetaLower), getValueAt(phiHigher, thetaLower), alphaPhi);
		fReal higherBelt = KaminoLerp<fReal>(getValueAt(phiLower, thetaHigher), getValueAt(phiHigher, thetaHigher), alphaPhi);

		fReal lerped = KaminoLerp<fReal>(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
}

fReal KaminoQuantity::getThetaOffset()
{
	return this->xOffset;
}

fReal KaminoQuantity::getPhiOffset()
{
	return this->yOffset;
}