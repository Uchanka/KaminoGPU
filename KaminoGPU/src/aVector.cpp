#include "aVector.h"
#include <algorithm>

enum { VX, VY, VZ, VW };
#pragma warning(disable : 4244)

// CONSTRUCTORS

vec3::vec3() 
{
    n[VX] = 0; n[VY] = 0; n[VZ] = 0; 
}

vec3::vec3(double x, double y, double z)
{ 
    n[VX] = x; n[VY] = y; n[VZ] = z; 
}

vec3::vec3(double d)
{ 
    n[VX] = n[VY] = n[VZ] = d; 
}

vec3::vec3(const vec3& v)
{ 
    n[VX] = v.n[VX]; n[VY] = v.n[VY]; n[VZ] = v.n[VZ]; 
}

// ASSIGNMENT OPERATORS

vec3& vec3::operator = (const vec3& v)
{ 
    n[VX] = v.n[VX]; n[VY] = v.n[VY]; n[VZ] = v.n[VZ]; return *this; 
}

vec3& vec3::operator += ( const vec3& v )
{ 
    n[VX] += v.n[VX]; n[VY] += v.n[VY]; n[VZ] += v.n[VZ]; return *this; 
}

vec3& vec3::operator -= ( const vec3& v )
{ 
    n[VX] -= v.n[VX]; n[VY] -= v.n[VY]; n[VZ] -= v.n[VZ]; return *this; 
}

vec3& vec3::operator *= ( double d )
{ 
    n[VX] *= d; n[VY] *= d; n[VZ] *= d; return *this; 
}

vec3& vec3::operator /= ( double d )
{ 
    double d_inv = 1.0f/d; n[VX] *= d_inv; n[VY] *= d_inv; n[VZ] *= d_inv;
    return *this; 
}

double& vec3::operator [] ( int i) {
    assert(! (i < VX || i > VZ));
    return n[i];
}

double vec3::operator [] ( int i) const {
    assert(! (i < VX || i > VZ));
    return n[i];
}

void vec3::set(double x, double y, double z)
{
   n[0] = x; n[1] = y; n[2] = z;
}

// SPECIAL FUNCTIONS

double vec3::Length() const
{  
    return sqrt(SqrLength()); 
}

double vec3::SqrLength() const
{  
    return n[VX]*n[VX] + n[VY]*n[VY] + n[VZ]*n[VZ]; 
}

vec3& vec3::Normalize() // it is up to caller to avoid divide-by-zero
{ 
    double len = Length();
    if (len > 0.000001) *this /= Length(); 
    return *this;
}

vec3 vec3::Cross(const vec3 &v) const
{
    vec3 tmp;
    tmp[0] = n[1] * v.n[2] - n[2] * v.n[1];
    tmp[1] = n[2] * v.n[0] - n[0] * v.n[2];
    tmp[2] = n[0] * v.n[1] - n[1] * v.n[0];
    return tmp;
}

void vec3::Print(const char* title) const
{
   printf("%s (%.4f, %.4f, %.4f)\n", title, n[0], n[1], n[2]);
}

// FRIENDS

vec3 operator - (const vec3& a)
{  
    return vec3(-a.n[VX],-a.n[VY],-a.n[VZ]); 
}

 vec3 operator + (const vec3& a, const vec3& b)
{ 
    return vec3(a.n[VX]+ b.n[VX], a.n[VY] + b.n[VY], a.n[VZ] + b.n[VZ]); 
}

 vec3 operator - (const vec3& a, const vec3& b)
{ 
    return vec3(a.n[VX]-b.n[VX], a.n[VY]-b.n[VY], a.n[VZ]-b.n[VZ]); 
}

 vec3 operator * (const vec3& a, double d)
{ 
    return vec3(d*a.n[VX], d*a.n[VY], d*a.n[VZ]); 
}

 vec3 operator * (double d, const vec3& a)
{ 
    return a*d; 
}

 double operator * (const vec3& a, const vec3& b)
{ 
    return (a.n[VX]*b.n[VX] + a.n[VY]*b.n[VY] + a.n[VZ]*b.n[VZ]); 
}

 vec3 operator / (const vec3& a, double d)
{ 
    double d_inv = 1.0f/d; 
    return vec3(a.n[VX]*d_inv, a.n[VY]*d_inv, a.n[VZ]*d_inv); 
}

 vec3 operator ^ (const vec3& a, const vec3& b)
{
    return vec3(a.n[VY]*b.n[VZ] - a.n[VZ]*b.n[VY],
        a.n[VZ]*b.n[VX] - a.n[VX]*b.n[VZ],
        a.n[VX]*b.n[VY] - a.n[VY]*b.n[VX]);
}

 int operator == (const vec3& a, const vec3& b)
{ 
    return (a.n[VX] == b.n[VX]) && (a.n[VY] == b.n[VY]) && (a.n[VZ] == b.n[VZ]);
}

 int operator != (const vec3& a, const vec3& b)
{ 
    return !(a == b); 
}

 void Swap(vec3& a, vec3& b)
{ 
    vec3 tmp(a); a = b; b = tmp; 
}

 vec3 Min(const vec3& a, const vec3& b)
{ 
    return vec3(std::min(a.n[VX], b.n[VX]), std::min(a.n[VY], b.n[VY]), std::min(a.n[VZ], b.n[VZ]));
}

 vec3 Max(const vec3& a, const vec3& b)
{ 
    return vec3(std::max(a.n[VX], b.n[VX]), std::max(a.n[VY], b.n[VY]), std::max(a.n[VZ], b.n[VZ]));
}

 vec3 Prod(const vec3& a, const vec3& b)
{ 
    return vec3(a.n[VX] * b.n[VX], a.n[VY] * b.n[VY], a.n[VZ] * b.n[VZ]); 
}

 double Dot(const vec3& a, const vec3& b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

 double Distance(const vec3& a, const vec3& b)  // distance
{
   return sqrt( (b[0]-a[0])*(b[0]-a[0]) +
                (b[1]-a[1])*(b[1]-a[1]) +
                (b[2]-a[2])*(b[2]-a[2]));
}

 double DistanceSqr(const vec3& a, const vec3& b)  // distance
{
   return ( (b[0]-a[0])*(b[0]-a[0]) +
            (b[1]-a[1])*(b[1]-a[1]) +
            (b[2]-a[2])*(b[2]-a[2]));
}

 double AngleBetween(const vec3& a, const vec3& b) // returns angle in radians
{
    // U.V = |U|*|V|*cos(angle)
    // angle = inverse cos (U.V/(|U|*|V|))
    double result = Dot(a,b) / (a.Length() * b.Length());
    result = std::min<double>(1.0,std::max<double>(-1.0,result));
    double radians = acos(result);
    return radians;
}

 std::istream& operator>>(std::istream& s, vec3& v)
{
    double x, y, z;
    s >> x >> y >> z;
    v = vec3(x, y, z);
    return s;
}

 std::ostream& operator<<(std::ostream& s, const vec3& v)
{
    s << (float) v[VX] << " " << (float) v[VY] << " " << (float) v[VZ];
    return s;
}

