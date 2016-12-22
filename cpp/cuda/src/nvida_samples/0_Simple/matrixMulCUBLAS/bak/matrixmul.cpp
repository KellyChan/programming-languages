/*

*/


#include <stdio.h>
#include <cstring>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif

#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif



// optional command-line multiplier for matrix sizes
typedef struct _matrixSize
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;


// Allocate a matrix with random float entries
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // by default, we use device 0
    // otherwise, we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        error = cudaSetDevice(devID);

        if (error != cudaSuccess)
        {
            printf("cudaSetDevice returned error code %d, liine(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    }

    iSizeMultiple = min(iSizeMultiple, 10);
    iSizeMultiple = max(iSizeMultiple, 1);

    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    } 

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name,
                                                                             deviceProp.major,
                                                                             deviceProp.minor);
    
    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;
    printf("block_size: %u, iSizeMultiple: %u\n", block_size, iSizeMultiple);

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if (matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
        printf("ERROR: Matrix sizes do not match!\n");
        exit(-1);
    }
}


int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    // set seed for rand()
    srand(2006);   

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float*)malloc(mem_size_A);

    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float*)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);
    
    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    // allocate host memory for the result
    float *h_C = (float *)malloc(mem_size_C);
    float *h_CUBLAS = (float *)malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(d_B, h_B, mem_size_B, cudaMemcpyHostTODevice));
    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 30;
    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        // Perform warmup operation with cublas
        checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB,
                                                                      matrix_size.uiHA,
                                                                      matrix_size.uiWA,
                                                                      &alpha,
                                                                      matrix_size.uiWB,
                                                                      d_A,
                                                                      matrix_size.uiWA,
                                                                      &beta,
                                                                      d_C,
                                                                      matrix_size.uiWB));
    }
}


int main(int argc, char **argv)
{
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    int devID = 0;
    int sizeMult = 5;
    sMatrixSize matrix_size;
    initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

    int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

    return matrix_result;
}
