/*
    cublasCopy - copy vector into vector

    - copy the vector x into the vector y
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
    printf("x: ");
    for (j = 0; j < n; j++)
    {
        printf("%2.0f,", x[j]);
    }
    printf("\n");
    y = (float*)malloc(n*sizeof(*y));

    // on the device
    float* d_x;
    float* d_y;
    cudaStat = cudaMalloc((void**)&d_x, n*sizeof(*x));
    cudaStat = cudaMalloc((void**)&d_y, n*sizeof(*y));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
    stat = cublasScopy(handle, n, d_x, 1, d_y, 1);
    stat = cublasGetVector(n, sizeof(float), d_y, 1, y, 1);
    
    printf("y after copy: \n");
    for (j = 0; j < n; j++)
    {
        printf("%2.0f,", y[j]);
    }
    printf("\n");

    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);

    free(x);
    free(y);
    
    return EXIT_SUCCESS;
  
}
