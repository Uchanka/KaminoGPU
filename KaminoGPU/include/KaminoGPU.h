# pragma once

# include "KaminoSolver.h"
# include <opencv2/opencv.hpp>

using namespace cv;

class Kamino
{
private:
	size_t nTheta;              // number of grid cells in u direction
	size_t nPhi;                // number of grid cells in v direction
	fReal gridLen;              // grid spacing (square in uv plane)
	fReal radius;               // radius of sphere

								/* practical condition: dt <= 5*dx / u_max */
								/* dt should be less than DT as well */
	float dt;                   // time step
	float DT;                   // frame rate @24 fps = 0.0147
	int frames;                 // number of frames to export
	fReal particleDensity;      // number of particles per unit area on the flat sphere

	std::string gridPath;       // folder destination for test bgeo files
	std::string particlePath;   // folder destination for simulation bgeo files

	std::string densityImage;   // file path of density image map
	std::string solidImage;     // file path of SOLIDCELL image map
	std::string colorImage;     // file path of image defining particle color

								/* velocity initialization */
								/* u = A + sin(B*phi / 2) * sin(C*theta / 2) */
								/* v = sin(D*phi / 2) * sin(E*theta / 2) */
								/* coefficients are for Fourier sums representing each of the above functions */
	fReal A;
	int B, C, D, E;

	Eigen::Matrix<fReal, 3, 1>* colorMap;

	/* initialize density field in solver */
	void initializeDensity(KaminoQuantity* d);
	/* define the locations of SOLID and FLUID grid cells */
	void defineCellTypes(gridType* g);
	/* load color image for particle color initialization */
	void loadColorImage();

	//size_t getIndex(size_t x, size_t y);

public:
	Kamino(fReal radius = 5.0, size_t nTheta = 128, fReal particleDensity = 200.0,
		float dt = 0.005, float DT = 1.0 / 24.0, int frames = 1000,
		fReal A = 0.0, int B = 1, int C = 1, int D = 1, int E = 1,
		std::string gridPath = "output/frame", std::string particlePath = "particles/frame",
		std::string densityImage = "", std::string solidImage = "", std::string colorImage = "");
	~Kamino();

	/* run the solver */
	void run();
};