/*
    cublasSdot - dot product

    This funciton computes the dot product of vectors x and y
             x * y = x0y0 + ... + xn-1yn-1

    for real vectors x and y
             x * y = x0y0_hat + ... + xn-1yn-1_hat

    for complex x, y.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define n 6


int main(void)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int j;
    float* x;
    float* y;

    x = (float*)malloc(n*sizeof(*x));
    for (j = 0; j < n; j++)
    {
        x[j] = (float)j;
    }

    y = (float*)malloc(n*sizeof(*y));
    for (j = 0; j < n; j++)
    {
        y[j] = (float)j;
    }

    printf("x, y:\n");
    for (j = 0; j < n; j++)
    {
        printf("%2.0f,", x[j]);
    }
    for (j = 0; j < n; j++)
    {
        printf("%2.0f,", y[j]);
    }
    printf("\n");

    // on the device
    float* d_x;
    float* d_y;
    cudaStat = cudaMalloc((void**)& d_x, n*sizeof(*x));
    cudaStat = cudaMalloc((void**)& d_y, n*sizeof(*y));
    
    stat = cublasCreate(&handle);
    stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
    stat = cublasSetVector(n, sizeof(*y), y, 1, d_y, 1);
    float result;
    stat = cublasSdot(handle, n, d_x, 1, d_y, 1, &result);

    printf("dot product x*y: \n");
    printf("%7.0f\n", result);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
    free(x);
    free(y);

    return EXIT_SUCCESS;
}
