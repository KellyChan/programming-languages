#include <stdio.h>
#include <cublas_v2.h>
#include <time.h>
#include <sys/time.h>

#define uS_PER_SEC 1000000
#define uS_PER_mS 1000

#define N_ROWS  1000
#define N_COLS 1000


int main()
{
    timeval t1, t2;

    float *A = (float*)malloc(N_ROWS * N_COLS * sizeof(float));

    // CPU
    gettimeofday(&t1, NULL);
    float *A_T = (float*)malloc(N_ROWS * N_COLS * sizeof(float));
    for (int i = 0; i < N_ROWS; i++)
    {
        for (int j = 0; j < N_COLS; j++)
        {
            A_T[(N_ROWS * j) + N_ROWS] = A[(i * N_COLS) + N_COLS];
        }
    }
    gettimeofday(&t2, NULL);
    float et1 = (((t2.tv_sec * uS_PER_SEC) + t2.tv_usec) - ((t1.tv_sec * uS_PER_SEC) + t1.tv_usec)) / (float)uS_PER_mS;
    printf("CPU time = %fms\n", et1);


    // GPU
    float *d_A, *d_A_T;
    float *h_A_T = (float*)malloc(N_ROWS * N_COLS * sizeof(float));
    cudaMalloc((void**)&d_A, N_ROWS * N_COLS * sizeof(float));
    cudaMalloc((void**)&d_A_T, N_ROWS * N_COLS * sizeof(float));
    cudaMemcpy(d_A, A, N_ROWS * N_COLS * sizeof(float), cudaMemcpyHostToDevice);

    gettimeofday(&t1, NULL);
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    gettimeofday(&t1, NULL);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N_ROWS, N_COLS, &alpha, d_A, N_COLS, &beta, d_A, N_ROWS, d_A_T, N_ROWS);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    cublasDestroy(handle);

    float et2 = (((t2.tv_sec*uS_PER_SEC) + t2.tv_usec) - ((t1.tv_sec * uS_PER_SEC) + t1.tv_usec)) / (float)uS_PER_mS;
    printf("GPU time = %fms\n", et2);

    cudaMemcpy(h_A_T, d_A_T, N_ROWS * N_COLS * sizeof(float), cudaMemcpyDeviceToHost);
     
    cudaFree(d_A);
    cudaFree(d_A_T);

    return 0;
}

