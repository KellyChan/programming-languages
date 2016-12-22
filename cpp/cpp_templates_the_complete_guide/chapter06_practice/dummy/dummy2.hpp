#ifndef DUMMY2_HPP
#define DUMMY2_HPP

// use export if USE_EXPORT is defined
#if define(USE_EXPORT)
#define EXPORT export
#else
#define EXPORT
#endif

// declaration of template
EXPORT
template <typename T>
void print_typeof(T const&);

// inlcude definition if USE_EXPORT is not defined
#if !define(USE_EXPORT)
#include "dummy.cpp"
#endif

#endif  // DUMMY2_HPP
