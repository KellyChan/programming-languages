/*

    cublasSaxpy: compute ax + y

        y = ax + y

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

    x = (float*)malloc (n*sizeof(*x));
    for (j = 0; j < n; j ++)
    {
        x[j] = (float)j;
    }
    
    y = (float*)malloc (n*sizeof(*y));
    for (j = 0; j < n; j++)
    {
        y[j] = (float)j;
    }
    printf("x, j:\n");

    for (j = 0; j < n; j++)
    {
        printf("%2.0f,", x[j]);
    }
    printf("\n");

    // on the device
    float* d_x;
    float* d_y;

    cudaStat = cudaMalloc((void**)&d_x, n*sizeof(*x));
    cudaStat = cudaMalloc((void**)&d_y, n*sizeof(*y));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
    stat = cublasSetVector(n, sizeof(*y), y, 1, d_y, 1);

    float a1 = 2.0;
    stat = cublasSaxpy(handle, n, &a1, d_x, 1, d_y, 1);
    stat = cublasGetVector(n, sizeof(float), d_y, 1, y, 1);
    printf("y after Saxpy: \n");
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
