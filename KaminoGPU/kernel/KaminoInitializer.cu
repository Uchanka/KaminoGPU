# include "../include/KaminoSolver.h"

void KaminoSolver::initialize_velocity()
{
	KaminoQuantity* u = this->velPhi;
	KaminoQuantity* v = this->velTheta;

	for (size_t j = 0; j < u->getNTheta(); ++j)
	{
		for (size_t i = 1; i < u->getNPhi(); ++i)
		{
			fReal ur_x = i * gridLen + gridLen / 2;
			fReal ur_y = (j + 1) * gridLen;
			fReal lr_x = i * gridLen + gridLen / 2;
			fReal lr_y = j * gridLen;
			fReal ul_x = i * gridLen - gridLen / 2;
			fReal ul_y = (j + 1) * gridLen;
			fReal ll_x = i * gridLen - gridLen / 2;
			fReal ll_y = j * gridLen;
			fReal noise_ur = FBM(ur_x, ur_y);
			fReal noise_lr = FBM(lr_x, lr_y);
			fReal noise_ul = FBM(ul_x, ul_y);
			fReal noise_ll = FBM(ll_x, ll_y);
			fReal noiseDy_l = (noise_ur - noise_lr);
			fReal noiseDy_r = (noise_ul - noise_ll);
			//fReal noiseDy_l = (noise_ur - noise_lr) / (radius * gridLen);
			//fReal noiseDy_r = (noise_ul - noise_ll) / (radius * gridLen);
			fReal avgNoise = (noiseDy_l + noiseDy_r) / 2.0;
			u->setCPUValueAt(i, j, avgNoise);
		}
	}
	// phi = 0 seam
	for (size_t j = 0; j < u->getNTheta(); ++j)
	{
		fReal ur_x = gridLen / 2;
		fReal ur_y = (j + 1) * gridLen;
		fReal lr_x = gridLen / 2;
		fReal lr_y = j * gridLen;
		fReal ul_x = 2 * M_PI - gridLen / 2;
		fReal ul_y = (j + 1) * gridLen;
		fReal ll_x = 2 * M_PI - gridLen / 2;
		fReal ll_y = j * gridLen;
		fReal noise_ur = FBM(ur_x, ur_y);
		fReal noise_lr = FBM(lr_x, lr_y);
		fReal noise_ul = FBM(ul_x, ul_y);
		fReal noise_ll = FBM(ll_x, ll_y);
		fReal noiseDy_l = (noise_ur - noise_lr);
		fReal noiseDy_r = (noise_ul - noise_ll);
		//fReal noiseDy_l = (noise_ur - noise_lr) / (radius * gridLen);
		//fReal noiseDy_r = (noise_ul - noise_ll) / (radius * gridLen);
		fReal avgNoise = (noiseDy_l + noiseDy_r) / 2.0;
		u->setCPUValueAt(0, j, avgNoise);
	}

	// set u_theta initial values using FBM curl noise
	for (size_t j = 0; j < v->getNTheta(); ++j)
	{
		for (size_t i = 0; i < v->getNPhi(); ++i)
		{
			fReal ur_x = (i + 1) * gridLen;
			fReal ur_y = j * gridLen + gridLen / 2;
			fReal lr_x = (i + 1) * gridLen;
			fReal lr_y = j * gridLen - gridLen / 2;
			fReal ul_x = i * gridLen;
			fReal ul_y = j * gridLen + gridLen / 2;
			fReal ll_x = i * gridLen;
			fReal ll_y = j * gridLen + gridLen / 2;
			fReal noise_ur = FBM(ur_x, ur_y);
			fReal noise_lr = FBM(lr_x, lr_y);
			fReal noise_ul = FBM(ul_x, ul_y);
			fReal noise_ll = FBM(ll_x, ll_y);
			fReal noiseDy_u = -1 * (noise_ur - noise_ul);
			fReal noiseDy_d = -1 * (noise_lr - noise_ll);
			//fReal noiseDy_u = -1 * (noise_ur - noise_ul) / (radius * gridLen * sin(j * gridLen + gridLen / 2));
			//fReal noiseDy_d = -1 * (noise_lr - noise_ll) / (radius * gridLen * sin(j * gridLen - gridLen / 2));
			fReal avgNoise = (noiseDy_u + noiseDy_d) / 2.0;
			v->setCPUValueAt(i, j, avgNoise);
		}
	}
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

	return 20.0 * total;
}

fReal kaminoLerpHost(fReal from, fReal to, fReal alpha)
{
	return (1.0 - alpha) * from + alpha * to;
}

fReal KaminoSolver::interpNoise2D(const fReal x, const fReal y) const
{
	fReal intX = std::floor(x);
	fReal fractX = x - intX;
	fReal intY = std::floor(y);
	fReal fractY = y - intY;

	fReal v1 = rand(vec2(intX, intY));
	fReal v2 = rand(vec2(intX + 1, intY));
	fReal v3 = rand(vec2(intX, intY + 1));
	fReal v4 = rand(vec2(intX + 1, intY + 1));

	// interpolate for smooth transitions
	fReal i1 = kaminoLerpHost(v1, v2, fractX);
	fReal i2 = kaminoLerpHost(v3, v4, fractX);
	return kaminoLerpHost(i1, i2, fractY);
}

fReal KaminoSolver::rand(const vec2 vecA) const
{
	// return pseudorandom number between -1 and 1
	vec2 vecB = vec2(12.9898, 4.1414);
	fReal dotProd = vecA[0] * vecB[0] + vecA[1] * vecB[1];
	fReal val = sin(dotProd * 43758.5453);
	return val - std::floor(val);
}