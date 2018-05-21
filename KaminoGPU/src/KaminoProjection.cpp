# include "../include/KaminoQuantity.h"

void KaminoSolver::projection()
{
	KaminoQuantity* u = staggeredAttr["u"];
	KaminoQuantity* v = staggeredAttr["v"];
	KaminoQuantity* p = centeredAttr["p"];

	/// TODO: Fill these divergence
	fillDivergence();
	/// TODO: Perform forward FFT on fourierF to make them fourier coefficients
	transformDivergence();

/*# ifdef OMParallelize
# pragma omp parallel for
# endif*/
	for (int nIndex = 0; nIndex < nPhi; ++nIndex)
	{
		int n = nIndex - nPhi / 2;

		for (int i = 0; i < nTheta; ++i)
		{
			fReal thetaI = (i + 0.5) * gridLen;

			b[i] = -2.0 / (gridLen * gridLen) - n * n / (sin(thetaI) * sin(thetaI));
			a[i] = 1.0 / (gridLen * gridLen) - cos(thetaI) / 2.0 / gridLen / sin(thetaI);
			c[i] = 1.0 / (gridLen * gridLen) + cos(thetaI) / 2.0 / gridLen / sin(thetaI);

			dReal[i] = this->fourieredFReal[getIndex(nIndex, i)];

			if (i == 0)
			{
				fReal coef = std::pow(-1.0, n);
				b[i] += coef * a[i];
				a[i] = 0.0;
			}
			if (i == nTheta - 1)
			{
				fReal coef = std::pow(-1.0, n);
				b[i] += coef * c[i];
				c[i] = 0.0;
			}
		}
		//When n == 0, d = 0, whole system degenerates to Ax = 0 where A is singular
		if (n != 0)
		{
			TDMSolve(this->a, this->b, this->c, this->dReal);
		}
		//d now contains Ui
		for (size_t UiIndex = 0; UiIndex < nTheta; ++UiIndex)
		{
			this->fourierUReal[getIndex(nIndex, UiIndex)] = dReal[UiIndex];
		}



		for (int i = 0; i < nTheta; ++i)
		{
			fReal thetaI = (i + 0.5) * gridLen;

			b[i] = -2.0 / (gridLen * gridLen) - n * n / (sin(thetaI) * sin(thetaI));
			a[i] = 1.0 / (gridLen * gridLen) - cos(thetaI) / 2.0 / gridLen / sin(thetaI);
			c[i] = 1.0 / (gridLen * gridLen) + cos(thetaI) / 2.0 / gridLen / sin(thetaI);

			dImag[i] = this->fourieredFImag[getIndex(nIndex, i)];

			if (i == 0)
			{
				fReal coef = std::pow(-1.0, n);
				b[i] += coef * a[i];
				a[i] = 0.0;
			}
			if (i == nTheta - 1)
			{
				fReal coef = std::pow(-1.0, n);
				b[i] += coef * c[i];
				c[i] = 0.0;
			}
		}
		//When n == 0, d = 0, whole system degenerates to Ax = 0 where A is singular
		if (n != 0)
		{
			TDMSolve(this->a, this->b, this->c, this->dImag);
		}
		//d now contains Ui
		for (size_t UiIndex = 0; UiIndex < nTheta; ++UiIndex)
		{
			this->fourierUImag[getIndex(nIndex, UiIndex)] = dImag[UiIndex];
		}
	}

	invTransformPressure();
	p->swapBuffer();

	fReal factorTheta = -invGridLen;
/*# ifdef OMParallelize
# pragma omp parallel for
# endif*/
	// Update velocities accordingly: uPhi
	for (size_t j = 0; j < u->getNTheta(); ++j)
	{
		for (size_t i = 0; i < u->getNPhi(); ++i)
		{
			fReal uBefore = u->getValueAt(i, j);
			fReal thetaBelt = (j + 0.5) * gridLen;
			fReal invSine = 1.0 / std::sin(thetaBelt);
			fReal factorPhi = factorTheta * invSine;

			size_t gridLeftI = (i == 0 ? u->getNPhi() - 1 : i - 1);
			size_t gridRightI = i;

			if (getGridTypeAt(gridLeftI, j) == SOLIDGRID ||
				getGridTypeAt(gridRightI, j) == SOLIDGRID)
			{
				u->writeValueTo(i, j, uSolid);
			}
			else
			{
				fReal pressurePhi = 0.0;
				if (getGridTypeAt(gridLeftI, j) == FLUIDGRID)
				pressurePhi -= p->getValueAt(gridLeftI, j);
				if (getGridTypeAt(gridRightI, j) == FLUIDGRID)
					pressurePhi += p->getValueAt(gridRightI, j);
				fReal deltauPhi = factorPhi * pressurePhi;
				u->writeValueTo(i, j, uBefore + deltauPhi);
			}
		}
	}

	u->swapBuffer();

/*# ifdef OMParallelize
# pragma omp parallel for
# endif*/
	// Update velocities accordingly: uTheta
	for (size_t j = 1; j < v->getNTheta() - 1; ++j)
	{
		for (size_t i = 0; i < v->getNPhi(); ++i)
		{
			fReal vBefore = v->getValueAt(i, j);
			size_t gridAboveJ = j;
			size_t gridBelowJ = j - 1;
			
			if (getGridTypeAt(i, gridBelowJ) == SOLIDGRID ||
				getGridTypeAt(i, gridAboveJ) == SOLIDGRID)
			{
				v->writeValueTo(i, j, vSolid);
			}
			else
			{
				fReal pressureTheta = 0.0;
				if (getGridTypeAt(i, gridBelowJ) == FLUIDGRID)
					pressureTheta -= p->getValueAt(i, gridBelowJ);
				if (getGridTypeAt(i, gridAboveJ) == FLUIDGRID)
					pressureTheta += p->getValueAt(i, gridAboveJ);
				fReal deltauTheta = factorTheta * pressureTheta;
				v->writeValueTo(i, j, deltauTheta + vBefore);
			}
		}
	}

	solvePolarVelocities();
	v->swapBuffer();

	//fillDivergence();
}
