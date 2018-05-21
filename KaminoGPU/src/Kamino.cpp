# include "../include/KaminoGPU.h"

Kamino::Kamino(fReal radius, size_t nTheta, fReal particleDensity,
        float dt, float DT, int frames,
		fReal A, int B, int C, int D, int E,
        std::string gridPath, std::string particlePath,
        std::string densityImage, std::string solidImage, std::string colorImage) :
        radius(radius), nTheta(nTheta), nPhi(2 * nTheta), gridLen(M_PI / nTheta),
        particleDensity(particleDensity),
        dt(dt), DT(DT), frames(frames),
		A(A), B(B), C(C), D(D), E(E),
        gridPath(gridPath), particlePath(particlePath),
        densityImage(densityImage), solidImage(solidImage), colorImage(colorImage)
{
	// stores BGR pixel values for an image
	// all values initialized to WHITE
	size_t size = nTheta * 2 * nTheta;
    this->colorMap = new Eigen::Matrix<fReal, 3, 1>[size];
	for (int i = 0; i < size; ++i) {
		colorMap[i] = Eigen::Matrix<fReal, 3, 1>(128.0, 128.0, 128.0);
	}

# ifdef OMParallelize
	omp_set_num_threads(TOTALThreads);
	Eigen::setNbThreads(TOTALThreads);
# endif

}

Kamino::~Kamino()
{
}

void Kamino::run()
{
    KaminoSolver solver(nPhi, nTheta, radius, gridLen, dt, A, B, C, D, E);
    KaminoQuantity* d = solver.getAttributeNamed("density");
    initializeDensity(d);
    gridType* g = solver.getGridTypeHandle();
    defineCellTypes(g);
	loadColorImage();
   
    KaminoParticles particles(particleDensity, radius, gridLen, &solver, nPhi, nTheta, densityImage, colorMap);
    KaminoQuantity* u = solver.getAttributeNamed("u");
    KaminoQuantity* v = solver.getAttributeNamed("v");

    solver.write_data_bgeo(gridPath, 0);
    particles.write_data_bgeo(particlePath, 0);

    float T = 0.0;              // simulation time
    for(int i = 1; i <= frames; i++){
        while(T < i*DT){
            solver.stepForward(dt);
            particles.updatePositions(u, v, dt);
            T += dt;
        }
        solver.stepForward(dt + i*DT - T);
        particles.updatePositions(u, v, dt);
        T = i*DT;

        solver.write_data_bgeo(gridPath, i);
        particles.write_data_bgeo(particlePath, i);
    }
}

void Kamino::loadColorImage()
{
    // read in image
    Mat image_in;
    image_in = imread(colorImage, IMREAD_COLOR);
    if(!image_in.data)
    {
        std::cout << "No color image provided. Particle color initialized to WHITE" << std::endl;
        return;
    }

	Mat image_flipped;
	cv::flip(image_in, image_flipped, 1);

    // resize to Nphi x Ntheta
    Mat image_sized;
    Size size(nPhi, nTheta);
    resize(image_flipped, image_sized, size);
    for(size_t i = 0; i < nPhi; ++i)
    {
        for(size_t j = 0; j < nTheta; ++j)
        {
            Point3_<uchar>* p = image_sized.ptr<Point3_<uchar>>(j, i);
            colorMap[getIndex(i, j)][2] = p->x / 255.0; // B
            colorMap[getIndex(i, j)][1] = p->y / 255.0; // G
            colorMap[getIndex(i, j)][0] = p->z / 255.0; // R
        }
    }
}

void Kamino::initializeDensity(KaminoQuantity* d)
{
	// read in image
	Mat image_in;
	image_in = imread(densityImage, IMREAD_COLOR);
	if (!image_in.data)
	{
		std::cout << "No density image provided. All density values initialized to ZERO." << std::endl;
		return;
	}
	Mat image_flipped;
	cv::flip(image_in, image_flipped, 1);

	// convert to greyscale
	Mat image_gray;
	cvtColor(image_flipped, image_gray, COLOR_BGR2GRAY);

	// resize to Nphi x Ntheta
	Mat image_sized;
	Size size(nPhi, nTheta);
	resize(image_gray, image_sized, size);

	for (size_t i = 0; i < nPhi; ++i)
	{
		for (size_t j = 0; j < nTheta; ++j)
		{
			Scalar intensity = image_sized.at<uchar>(Point(i, j));
			fReal scale = static_cast <fReal> (intensity.val[0]) / 255.0;
			d->setValueAt(i, j, scale);
		}
	}
}

void Kamino::defineCellTypes(gridType* g)
{
    for (size_t gPhi = 0; gPhi < nPhi; ++gPhi)
    {
        for (size_t gTheta = 0; gTheta < nTheta; ++gTheta)
        {
            *(g + getIndex(gPhi, gTheta)) = FLUIDGRID;
        }
    }

	// read in image
	Mat image_in;
	image_in = imread(solidImage, IMREAD_COLOR);
	if (!image_in.data)
	{
		std::cout << "No grid type image provided. All cells initialized to FLUID" << std::endl;
		return;
	}
	Mat image_flipped;
	cv::flip(image_in, image_flipped, 1);

	//convert to greyscale
	Mat image_gray;
	cvtColor(image_flipped, image_gray, COLOR_BGR2GRAY);

	// resize to Nphi x Ntheta
	Mat image_sized;
	Size size(nPhi, nTheta);
	resize(image_gray, image_sized, size);

	//define SOLID cells beneath some threshold
	for (size_t i = 0; i < nPhi; ++i)
	{
		for (size_t j = 0; j < nTheta; ++j)
		{
			Scalar intensity = image_sized.at<uchar>(Point(i, j));
			if (intensity.val[0] > 128) {
				*(g + getIndex(i, j)) = SOLIDGRID;
			}
		}
	}
}


size_t Kamino::getIndex(size_t x, size_t y)
{
    return y * nPhi + x;
}