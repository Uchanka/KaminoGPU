# include "../include/KaminoQuantity.h"
# include "../include/CubicSolver.h"

void KaminoSolver::geometric()
{
	KaminoQuantity* u = staggeredAttr["u"];
	KaminoQuantity* v = staggeredAttr["v"];

# ifdef OMParallelize
# pragma omp parallel for
# endif
	for (int thetaJ = 1; thetaJ < nTheta - 1; ++thetaJ)
	{
		for (size_t phiI = 0; phiI < nPhi; ++phiI)
		{
			fReal thetaAtJ = thetaJ * gridLen;
			fReal uPrev = u->getValueAt(phiI, thetaJ);
			fReal vPrev = v->getValueAt(phiI, thetaJ);

			fReal G = 0.0;
			fReal uNext = 0.0;
			if (std::abs(thetaAtJ - M_PI / 2.0) < 1e-8)
			{
				G = 0.0;
				uNext = uPrev;
			}
			else
			{
				G = timeStep * std::cos(thetaJ * gridLen) / (radius * sin(thetaJ * gridLen));
				fReal cof = G * G;
				fReal A = 0.0;
				fReal B = (G * vPrev + 1.0) / cof;
				fReal C = -uPrev / cof;

				fReal solution[3];
				SolveP3(solution, A, B, C);

				uNext = solution[0];
			}
			fReal vNext = vPrev + G * uNext * uNext;

			u->writeValueTo(phiI, thetaJ, uNext);
			v->writeValueTo(phiI, thetaJ, vNext);
		}
	}
	solvePolarVelocities();

	u->swapBuffer();
	v->swapBuffer();
}
