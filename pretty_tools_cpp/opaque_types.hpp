#ifndef __PRTTYTOOLS_OPAQUE_TYPES_H__
#define __PRTTYTOOLS_OPAQUE_TYPES_H__


#include <complex>
#include <map>
#include <vector>

// 这块是负责将c++的map和vector进行绑定，对性能较好，因为绑定后不必做不必要的转换

// include<pybind11/stl_bind.h>
// using MapStringComplex = std::map<std::string, std::complex<double>>;
// using VectorPairStringDouble = std::vector<std::pair<std::string, double>>;
// PYBIND11_MAKE_OPAQUE(MapStringComplex);
// PYBIND11_MAKE_OPAQUE(VectorPairStringDouble);


#endif // __OPAQUE_TYPES_H__