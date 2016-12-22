#include <cmath>
#include <iostream>

#include <omp.h>

using namespace std;


// create storage on the device in gmem
__device__ float d_a[32], d_d[32];
__device__ float d_e[32], d_f[32];

#define NUM_ITERATIONS (1024 * 1024);

#ifdef ILP4

    // test instruction level parallelism
    #define OP_COUNT 4 * 2 * NUM_ITERATIONS

    __global__ void kernel(float a, float b, float c)
    {
        register float d = a;
        register float e = a;
        register float f = a;
        #pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            a = a * b + c;
            d = d * b + c;
            e = e * b + c;
            f = f * b + c;
        }

        // write to gmem so the work is not optimized out by the compiler
        d_a[threadIdx.x] = a;
        d_d[threadIdx.x] = d;
        d_e[threadIdx.x] = e;
        d_f[threadIdx.x] = f;
    }

#else

    // test thread level parallelism
    #define OP_COUNT 1 * 2 * NUM_ITERATIONS

    __global__ void kernel(float a, float b, float c)
    {
        #pragma unroll 16
        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            a = a * b + c;
        }

        // write to gmem so the work is not optimized out by the compiler
        d_a[threadIdx.x] = a;
    }

#endif


int main()
{
    // iterate over number of threads in a block
    for (int nThreads = 32; nThreads <= 1024; nThreads += 32)
    {
        double start = omp_get_wtime();
        kernel<<<1, nThreads>>>(1., 2., 3.);  // async kernel launch

        if (cudaGetLastError() != cudaSuccess)
        {
            cerr << "Launch error" << endl;
            return (1);
        }

        cudaThreadSynchronize();  // need to wait for the kernel to complete
        double end = omp_get_wtime();
        cout << "warps" << ceil(nThreads/32) << " "
             << nThreads << " " << (nThreads * (OP_COUNT/1.e9) / (end - start))
             << "Gflops" << endl;
    }

    return (0);
}
