# ifndef VECUTIL_H
# define VECUTIL_H

# include <iostream>
# include <assert.h>

//enum components{x, y, z};

class vec2
{
public:

	double n[2];

	// Constructors
	vec2();
	vec2(double x, double y);
	vec2(double d);
	vec2(const vec2& v);					// copy constructor

	double& operator [] (int i);				// indexing
	double operator[] (int i) const;			// read-only indexing
	vec2& operator = (const vec2& v);	    // assignment of a vec3
};

class vec3
{
public:

	double n[3];

	// Constructors
	vec3();
	vec3(double x, double y, double z);
	vec3(double d);
	vec3(const vec3& v);					// copy constructor

	double& operator [] (int i);				// indexing
	double operator[] (int i) const;			// read-only indexing
	vec3& operator = (const vec3& v);	    // assignment of a vec3
};


#endif

