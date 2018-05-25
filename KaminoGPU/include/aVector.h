// Aline Normoyle 2015, based on code by Liming Zhao
//
#ifndef aVector_H_
#define aVector_H_

#include <iostream>
#include <assert.h>

class vec3
{
public:

    double n[3];

    // Constructors
    vec3();
    vec3(double x, double y, double z);
    vec3(double d);
    vec3(const vec3& v);					// copy constructor

    // Assignment operators
    vec3& operator = ( const vec3& v );	    // assignment of a vec3
    vec3& operator += ( const vec3& v );	    // incrementation by a vec3
    vec3& operator -= ( const vec3& v );	    // decrementation by a vec3
    vec3& operator *= ( double d );	    // multiplication by a constant
    vec3& operator /= ( double d );	    // division by a constant
    double& operator [] ( int i);				// indexing
    double operator[] (int i) const;			// read-only indexing

    // special functions
    double Length() const;				// length of a vec3
    double SqrLength() const;				// squared length of a vec3
    vec3& Normalize();					// normalize a vec3 in place
    vec3 Cross(const vec3 &v) const;			// cross product
    void Print(const char* title) const;
    void set(double x, double y, double z);
    
    // friends
    friend  vec3 operator - (const vec3& v);				// -v1
    friend  vec3 operator + (const vec3& a, const vec3& b);	    // v1 + v2
    friend  vec3 operator - (const vec3& a, const vec3& b);	    // v1 - v2
    friend  vec3 operator * (const vec3& a, double d);	    // v1 * 3.0
    friend  vec3 operator * (double d, const vec3& a);	    // 3.0 * v1
    friend  double operator * (const vec3& a, const vec3& b);    // dot product
    friend  vec3 operator / (const vec3& a, double d);	    // v1 / 3.0
    friend  vec3 operator ^ (const vec3& a, const vec3& b);	    // cross product
    friend  int operator == (const vec3& a, const vec3& b);	    // v1 == v2 ?
    friend  int operator != (const vec3& a, const vec3& b);	    // v1 != v2 ?

    friend  void Swap(vec3& a, vec3& b);						// swap v1 & v2
    friend  vec3 Min(const vec3& a, const vec3& b);		    // min(v1, v2)
    friend  vec3 Max(const vec3& a, const vec3& b);		    // max(v1, v2)
    friend  vec3 Prod(const vec3& a, const vec3& b);		    // term by term *
    friend  double Dot(const vec3& a, const vec3& b);			// dot product
    friend  double Distance(const vec3& a, const vec3& b);  // distance
    friend  double DistanceSqr(const vec3& a, const vec3& b);  // distance sqr
    friend  double AngleBetween(const vec3& a, const vec3& b); // returns angle in radians

    // input output
    friend  std::istream& operator>>(std::istream& s, vec3& v);
    friend  std::ostream& operator<<(std::ostream& s, const vec3& v);
};

const vec3 axisX(1.0f, 0.0f, 0.0f);
const vec3 axisY(0.0f, 1.0f, 0.0f);
const vec3 axisZ(0.0f, 0.0f, 1.0f);
const vec3 vec3Zero(0.0f, 0.0f, 0.0f);


#endif

