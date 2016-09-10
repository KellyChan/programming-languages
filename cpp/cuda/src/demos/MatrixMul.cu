#include <stdio.h>
#include <assert.h>
#include <cstdlib>

#include "../utils/utils.h"
#include "../utils/timer.h"


template <int BLOCK_SIZE>
__global__ void matrixMulCUDA (float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        
        // declaration of the shared memory array As used to store
        // the sub-matrix of B
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // declaration of the shared memory array Bs used to store
        // the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // load the matrices from device memory
        // to shared memory, each thread loads one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // synchronize to make sure the matrices are loaded
        __syncthreads();

        // multiply the two matrices together
        // each thread computes one element of the block sub-matrix
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // synchronize to make sure that the preceding computation
        // is donen before loading two new sub-matrices of A and B
        // in the next iteration
        __syncthreads();
    }

    // write the block sub-matrix to device memory
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


void constantInit (float *data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}


/*
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply (int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB)
{
    // allocate host memory for amtrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // allocate device memory
    float *d_A, *d_B, *d_C;

    // allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *)malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // performs warmup operation using matrixMul CUDA kernel
    matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    printf("done\n");
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    GpuTimer timer; timer.Start();
    // execute the kernel
    int nIter = 30;
    for (int j = 0; j < nIter; j++) {
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    // record the stop event
    timer.Stop();

    float msecTotal = timer.Elapsed();
    // compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    double effectiveBytes = (((double)(dimsA.x + dimsB.y) * dimsA.y * dimsB.x) + dimsA.y * dimsB.x) * sizeof(int);
    double effectiveBandwidth = effectiveBytes / 1E6 / msecPerMatrixMul;
    double bAx = dimsA.x / 32;
    double bAy = dimsA.y / 32;
    double bBx = dimsB.x / 32;
    double bBy = dimsB.y / 32;
    double blocksLoaded = ((bAx + bBy) * bAy * bBx) + bAy * bBx;
    double bytesLoaded = blocksLoaded * 32 * 32 * sizeof(int);
    double actualBandwidth = bytesLoaded / 1E6 / msecPerMatrixMul;
    printf(
        "Performace= %.2f GFlop/s, EFfective %.2f GBytes/s, Actual %.2f GBytes/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        effectiveBandwidth,
        actualBandwidth,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y
    );

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    printf("Checking computed result for correctness: ");
    bool correct = true;
    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
        if (fabs(h_C[i] - (dimsA.x * valB)) > 1e-3) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-4\n", i, h_C[i], dimsA.x * valB);
            correct = false;
        }
    }
    printf("%s\n", correct ? "OK" : "FAIL");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


/*
* Program main
*/
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (argc != 5) {
        printf("Usage: \n");
        printf("      WidthA HeightA (Width x Height of matrix A)\n");
        printf("      WidthB HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");
        exit(EXIT_SUCCESS);
   
    }

    // Use a larger block size for Fermi and above
    int block_size = 32;
    dim3 dimsA;
    dim3 dimsB;

    // width of Matrix A
    dimsA.x = atoi(argv[1]);
    // height of Matrix A
    dimsA.y = atoi(argv[2]);
    // width of Matrix B
    dimsB.x = atoi(argv[3]);
    // height of Matrix B
    dimsB.y = atoi(argv[4]);

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d, %d), MatrixB(%d, %d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);
    exit(matrix_result);
}
