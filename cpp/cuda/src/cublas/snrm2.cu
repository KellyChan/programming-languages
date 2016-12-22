/*
    cublasSnrm2 - Euclidean norm

    This function computes the Euclidean norm of the vector x
        |x| = sqrt(x0^2 + ... + x_(n-1)^2)

        where x = {x0,...,x_(n-1)}
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
    x = (float*)malloc(n*sizeof(*x));
    for (j = 0; j < n; j++)
    {
        x[j] = (float)j;
    }
    printf("x:");
    for (j = 0; j < n; j++)
    {
        printf("%2.0f", x[j]);
    }
    printf("\n");

    // on the device
    float* d_x;
    cudaStat = cudaMalloc((void**)& d_x, n*sizeof(*x));
    stat = cublasCreate(&handle);
    stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
    float result;
    stat = cublasSnrm2(handle, n, d_x, 1, &result);
    printf("Euclidean norm of x: ");
    printf("%7.3f\n", result);

    cudaFree(d_x);
    cublasDestroy(handle);
    free(x);
    return EXIT_SUCCESS;
}
