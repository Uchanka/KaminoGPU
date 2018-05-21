# include "../include/KaminoQuantity.h"

KaminoParticles::KaminoParticles(fReal particleDensity, fReal radius, fReal h, KaminoSolver* solver, size_t nPhi, size_t nTheta, std::string densityImage, Eigen::Matrix<fReal, 3, 1>* colorMap) :
                            particleDensity(particleDensity), radius(radius), parentSolver(solver)
{

    // default uniform particle initialization
    if(densityImage == ""){
        fReal linearDensity = sqrt(particleDensity);
        fReal delta = M_PI / nTheta / linearDensity;
        fReal halfDelta = delta / 2.0;

        unsigned int numThetaParticles = linearDensity * nTheta;
        unsigned int numPhiParticles = 2 * numThetaParticles;

        for(unsigned int i = 0; i < numPhiParticles; ++i){
            for(unsigned int j = 0; j < numThetaParticles; ++j){
                // distribute in phi and theta randomly
                // +/- is 50/50
                fReal signPhi = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
                signPhi = signPhi >= 0.5 ? 1.0 : -1.0;
                fReal signTheta = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
                signTheta = signTheta >= 0.5 ? 1.0 : -1.0;

                // get random value between 0 and halfDelta in +/- direction
                fReal randPhi = signPhi * halfDelta * static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
                fReal randTheta = signTheta * halfDelta * static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);

                // assign positions (phi, theta)
                fReal phi = i * delta + randPhi;
                fReal theta = j * delta + randTheta;
				if (phi < 0.0)
					phi = 0.0;
				if (theta < 0.0)
					theta = 0.0;
                
                // check to make sure particle isn't in a solid cell
                size_t x = std::floor(phi / h);
                size_t y = std::floor(theta / h);
                gridType* cellMap = solver->getGridTypeHandle();
                gridType type = *(cellMap + solver->getIndex(x, y));
                if(type == SOLIDGRID){
                    continue;
                }

				// set particle position
                Eigen::Matrix<fReal, 2, 1> pos(phi, theta);
                positions.push_back(pos);

                // initialize velocities (0,0)
                Eigen::Matrix<fReal, 2, 1> vel(0.0, 0.0);
                velocities.push_back(vel);

				// define particle color
				Eigen::Matrix<fReal, 3, 1> color = *(colorMap + solver->getIndex(x, y));
				colors.push_back(color);
            }
        }

    }
    // initialize particles according to density image file
    else{
        KaminoQuantity* d = solver->getAttributeNamed("density");
        for(size_t i = 0; i < nPhi; ++i)
        {
            for(size_t j = 0; j < nTheta; ++j)
            {
                fReal scale = d->getValueAt(i, j);
                fReal density = scale * particleDensity;
                fReal linearDensity = sqrt(density);
                fReal delta = M_PI / nTheta / linearDensity;
                fReal halfDelta = delta / 2.0;

                unsigned int numThetaParticles = linearDensity;
                unsigned int numPhiParticles = numThetaParticles;

                for(unsigned int m = 0; m < numPhiParticles; ++m){
                    for(unsigned int n = 0; n < numThetaParticles; ++n){
                        // distribute in phi and theta randomly
                        // +/- is 50/50
                        fReal signPhi = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
                        signPhi = signPhi >= 0.5 ? 1.0 : -1.0;
                        fReal signTheta = static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
                        signTheta = signTheta >= 0.5 ? 1.0 : -1.0;

                        // get random value between 0 and halfDelta in +/- direction
                        fReal randPhi = signPhi * halfDelta * static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);
                        fReal randTheta = signTheta * halfDelta * static_cast <fReal> (rand()) / static_cast <fReal> (RAND_MAX);

                        // assign positions (phi, theta)
                        fReal phi = i * h + m * delta + randPhi;
                        fReal theta = j * h + n * delta + randTheta;
						if (phi < 0.0)
							phi = 0.0;
						if (theta < 0.0)
							theta = 0.0;
                        
                        // check to make sure particle isn't in a solid cell
                        size_t x = std::floor(phi / h);
                        size_t y = std::floor(theta / h);
                        gridType* cellMap = solver->getGridTypeHandle();
                        gridType type = *(cellMap + solver->getIndex(x, y));
                        if(type == SOLIDGRID){
                            continue;
                        }

						// set particle position
                        Eigen::Matrix<fReal, 2, 1> pos(phi, theta);
                        positions.push_back(pos);

                        // initialize velocities (0,0)
                        Eigen::Matrix<fReal, 2, 1> vel(0.0, 0.0);
                        velocities.push_back(vel);        

						// define particle color
						Eigen::Matrix<fReal, 3, 1> color = *(colorMap + solver->getIndex(x, y));
						colors.push_back(color);
                    }
                }
            }
        }
    }
}

KaminoParticles::~KaminoParticles()
{
}

void KaminoParticles::updatePositions(KaminoQuantity* u, KaminoQuantity* v, fReal deltaT)
{
# ifdef OMParallelize
# pragma omp parallel for
# endif
    for(int i = 0; i < positions.size(); ++i){
        fReal uPhi = u->sampleAt(positions[i][0], positions[i][1], parentSolver->uNorthP, parentSolver->uSouthP);
        fReal uTheta = v->sampleAt(positions[i][0], positions[i][1], parentSolver->uNorthP, parentSolver->uSouthP);
        fReal nextPhi;
        fReal nextTheta;
        // problem at the south pole
        if(positions[i][1] > M_PI - 1E-7 && positions[i][1] < M_PI + 1E-7){
            nextPhi = positions[i][0] + uPhi * deltaT / (radius * sin(1E-10));
        }
        // problem at the north pole
        else if(positions[i][1] < 1E-7 && positions[i][1] > -1E-7){
            nextPhi = positions[i][0] + uPhi * deltaT / (radius * sin(1E-10));
        }
        else{
            nextPhi = positions[i][0] + uPhi * deltaT / (radius * sin(positions[i][1]));
        } 
        nextTheta = positions[i][1] + uTheta * deltaT / radius;
        // wrap particles
        bool check = validatePhiTheta(nextPhi, nextTheta);
        positions[i][0] = nextPhi;
        positions[i][1] = nextTheta;
    }
}

void KaminoParticles::updateVelocities(KaminoQuantity* u, KaminoQuantity* v)
{
    for(unsigned int i = 0; i < velocities.size(); ++i){
        fReal uPhi = u->sampleAt(positions[i][0], positions[i][1], parentSolver->uNorthP, parentSolver->uSouthP);
        fReal uTheta = v->sampleAt(positions[i][0], positions[i][1], parentSolver->uNorthP, parentSolver->uSouthP);
        velocities[i][0] = uPhi;
        velocities[i][1] = uTheta;        
    }
}

void KaminoParticles::write_data_bgeo(const std::string& s, const int frame)
{
    std::string file = s + std::to_string(frame) + ".bgeo";
    std::cout << "Writing to: " << file << std::endl;
    Partio::ParticlesDataMutable* parts = Partio::create();
	Partio::ParticleAttribute posH, vH, col;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    vH = parts->addAttribute("v", Partio::VECTOR, 3);
	col = parts->addAttribute("color", Partio::VECTOR, 3);

    for(unsigned int i = 0; i < positions.size(); ++i){
        int idx = parts->addParticle();
        float* p = parts->dataWrite<float>(posH, idx);
        float* v = parts->dataWrite<float>(vH, idx);
		float* c = parts->dataWrite<float>(col, idx);
        Eigen::Matrix<float, 3, 1> pos(positions[i][0], positions[i][1], 0.0);
        Eigen::Matrix<float, 3, 1> vel(velocities[i][0], positions[i][1], 0.0);
        mapVToSphere(pos, vel);
        mapPToSphere(pos);
        for (int k = 0; k < 3; ++k){
            p[k] = pos(k, 0);
            v[k] = vel(k, 0);
			c[k] = colors[i][k];
        }
    }
    Partio::write(file.c_str(), *parts);
    parts->release();
}

void KaminoParticles::mapPToSphere(Eigen::Matrix<float, 3, 1>& pos) const
{
    float theta = pos[1];
    float phi = pos[0];
    pos[0] = radius * sin(theta) * cos(phi);
    pos[2] = radius * sin(theta) * sin(phi);
    pos[1] = radius * cos(theta);
}

void KaminoParticles::mapVToSphere(Eigen::Matrix<float, 3, 1>& pos, Eigen::Matrix<float, 3, 1>& vel) const
{
    float theta = pos[1];
    float phi = pos[0];

    float u_theta = vel[1];
    float u_phi = vel[2];

    vel[0] = cos(theta) * cos(phi) * u_theta - sin(phi) * u_phi;
    vel[2] = cos(theta) * sin(phi) * u_theta + cos(phi) * u_phi;
    vel[1] = -sin(theta) * u_theta;
}