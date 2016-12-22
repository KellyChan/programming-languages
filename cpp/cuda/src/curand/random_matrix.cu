#include <ctime>
#include <iostream>

#include <curand.h>
#include <cublas_v2.h>

using namespace std;


void fill_rand_gpu(float *A, int A_rows, int A_cols)
{
    // create a pseudo-random number generator
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);

    // set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(curand_gen, (unsigned long long) clock());

    // fill the array with random numbers on the device
    curandGenerateUniform(curand_gen, A, A_rows * A_cols);
}


void cublas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
    int lda = m;
    int ldb = k;
    int ldc = m;

    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // create a handle for cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // destroy the handle
    cublasDestroy(handle);
}


// print matrix A(A_rows, A_cols) storage in column-major format
void print_matrix(const float *A, int A_rows, int A_cols)
{
    for (int i = 0; i < A_rows; ++i)
    {
        for (int j = 0; j < A_cols; ++j)
        {
            cout << A[A_rows * j + i] << "";
        }
        cout << endl;
    }
    cout << endl;
}


int main()
{
    // allocate 3 arrays on cpu
    int A_rows, A_cols;
    int B_rows, B_cols;
    int C_rows, C_cols;

    // sqaure arrays
    A_rows = A_cols = 3;
    B_rows = B_cols = 3;
    C_rows = C_cols = 3;

    float* h_A = (float*)malloc(A_rows * A_cols * sizeof(float));
    float* h_B = (float*)malloc(B_rows * B_cols * sizeof(float));
    float* h_C = (float*)malloc(C_rows * C_cols * sizeof(float));

    // allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_rows * A_cols * sizeof(float));
    cudaMalloc(&d_B, B_rows * B_cols * sizeof(float));
    cudaMalloc(&d_C, C_rows * C_cols * sizeof(float));

    // fill the arrays A and B on GPU with random numbers
    fill_rand_gpu(d_A, A_rows, A_cols);
    fill_rand_gpu(d_B, B_rows, B_cols);

    // optionally we can copy the data back on CPU and print the arrays
    cudaMemcpy(h_A, d_A, A_rows * A_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, B_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "A = " << endl;
    print_matrix(h_A, A_rows, A_cols);
    cout << "B = " << endl;
    print_matrix(h_B, B_rows, B_cols);

    // multiply A and B on GPU
    cublas_mmul(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    // copy and print the result on host memory
    cudaMemcpy(h_C, d_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "C = " << endl;
    print_matrix(h_C, C_rows, C_cols);

    // free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
