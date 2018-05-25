#pragma once

# include "KaminoSolver.h"

class KaminoParticles
{
private:
	fReal particleDensity;
	fReal radius;
	std::vector<vec2> positions;
	std::vector<vec2> velocities;
	std::vector<vec3> colors;
	KaminoSolver* parentSolver;

	void mapPToSphere(vec3& pos) const;
	void mapVToSphere(vec3& pos, vec3& vel) const;
public:
	KaminoParticles(fReal particleDensity, fReal radius, fReal h, KaminoSolver* parentSolver, size_t nPhi, size_t nTheta, std::string densityImage,
		vec3* colorMap);
	~KaminoParticles();

	void updatePositions(KaminoQuantity* u, KaminoQuantity* v, fReal deltaT);
	void updateVelocities(KaminoQuantity* u, KaminoQuantity* v);
	void write_data_bgeo(const std::string& s, const int frame);
};