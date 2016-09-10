#ifndef UTILS_H__
#define UTILS_H__


#include <iostream>


#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {

    if (err != cudaSuccess) {

        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);

    }

}

#endif
