/*
    matrix-matrix multiplcation

        C = alpha * op(A) * op(B) + beta * C
   
      - alpha/beta: scalar
      - A/B: matrix in column-major format
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i, j, ld) ( ((j)*(ld)) + (i))
#define m 6  // a - mxk matrix
#define n 4  // b - kxn matrix
#define k 5  // c - mxn matrix


int main(void)
{
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cublas function status
    cublasHandle_t handle; // cublas context

    int i, j;  // i - row index, j - col index
    float* a;  // mxk matrix a on the host
    float* b;  // kxn matrix b on the host
    float* c;  // mxn matrix c on the host
    a = (float*)malloc(m*k*sizeof(float));  // host memory for a
    b = (float*)malloc(k*n*sizeof(float));  // host memory for b
    c = (float*)malloc(m*n*sizeof(float));  // host memory for c
    
    // define an mxk matrix a column by column
    // a:
    // 11, 17, 23, 29, 35
    // 12, 18, 24, 30, 36
    // 13, 19, 25, 31, 37
    // 14, 20, 26, 32, 38
    // 15, 21, 27, 33, 39
    // 16, 22, 28, 34, 40
    int ind = 11;
    for (j = 0; j < k; j++)
    {
        for (i = 0; i < m; i++)
        {
            a[IDX2C(i, j, m)] = (float)ind++;
        }
    }

    // b
    ind = 11;
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < k; i++)
        {
            b[IDX2C(i, j, k)] = (float)ind++;
        }
    }

    // c
    ind = 11;
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < m; i++)
        {
            c[IDX2C(i, j, m)] = (float)ind++;
        }
    }


    // on the device
    float* d_a;
    float* d_b;
    float* d_c;
    cudaStat = cudaMalloc((void**)& d_a, m*k*sizeof(*a));
    cudaStat = cudaMalloc((void**)& d_b, k*n*sizeof(*b));
    cudaStat = cudaMalloc((void**)& d_c, m*n*sizeof(*c));

    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m);  // a -> d_a;
    stat = cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k);  // b -> d_b;
    stat = cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m);  // c -> d_c
    float alpha = 1.0f;
    float beta = 1.0f;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, m, d_b, k, &beta, d_c, m);
    stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m);  // cp d_c -> c
    printf("c after Sgemm: \n");
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%7.0f", c[IDX2C(i, j, m)]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
    free(a);
    free(b);
    free(c);
    
    return EXIT_SUCCESS;
}
