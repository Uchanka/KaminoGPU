#include "vectorUtil.cuh"
#include <algorithm>

enum { X, Y, Z, W };
#pragma warning(disable : 4244)

// CONSTRUCTORS

vec3::vec3()
{
	n[X] = 0;
	n[Y] = 0;
	n[Z] = 0;
}

vec3::vec3(double x, double y, double z)
{
	n[X] = x;
	n[Y] = y;
	n[Z] = z;
}

vec3::vec3(double d)
{
	n[X] = n[Y] = n[Z] = d;
}

vec3::vec3(const vec3& v)
{
	n[X] = v.n[X]; n[Y] = v.n[Y]; n[Z] = v.n[Z];
}

vec3& vec3::operator = (const vec3& v)
{
	n[X] = v.n[X]; n[Y] = v.n[Y]; n[Z] = v.n[Z];
	return *this;
}

double& vec3::operator [] (int i)
{
	assert(!(i < X || i > Z));
	return n[i];
}

double vec3::operator [] (int i) const
{
	assert(!(i < X || i > Z));
	return n[i];
}

vec2::vec2()
{
	n[X] = 0;
	n[Y] = 0;
}

vec2::vec2(double x, double y)
{
	n[X] = x;
	n[Y] = y;
}

vec2::vec2(double d)
{
	n[X] = n[Y] = d;
}

vec2::vec2(const vec2& v)
{
	n[X] = v.n[X];
	n[Y] = v.n[Y];
}

vec2& vec2::operator = (const vec2& v)
{
	n[X] = v.n[X];
	n[Y] = v.n[Y];
	return *this;
}

double& vec2::operator [] (int i)
{
	assert(!(i < X || i > Y));
	return n[i];
}

double vec2::operator [] (int i) const
{
	assert(!(i < X || i > Y));
	return n[i];
}
