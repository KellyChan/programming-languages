#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <vector>

#include "../utils/utils.h"


template<typename T>
__global__ void simpleTranspose (T *array_in, T *array_out, int rows_in, int cols_in) 
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    int col = tid % cols_in;
    int row = tid / cols_in;

    array_out[col * rows_in + row] = array_in[row * cols_in + col];
}


template<typename T>
__global__ void simpleTranspose2D (T *array_in, T *array_out, int rows_in, int cols_in) 
{
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;

    array_out[col * rows_in + row] = array_in[row * cols_in + col];
}


template<typename T, int numWarps>
__global__ void fastTranspose (T *array_in, T *array_out, int rows_in, int cols_in) 
{
    const int warpId = threadIdx.y;
    const int lane = threadIdx.x;
    const int warpSize = 32;

    __shared__ T block[warpSize][warpSize + 1];

    int bc = blockIdx.x;
    int br = blockIdx.y;

    // load 32*32 block into shared memory
    for (int i=0; i<warpSize/numWarps; ++i) {
        int gr = br * warpSize + i * numWarps + warpId;
        int gc = bc * warpSize + lane;

        block[i * numWarps + warpId][lane] = array_in[gr * cols_in + gc];
    }

    __syncthreads();

    // now we switch to each warp outputting a row, which will read
    // from a column in the shared memory
    // this way everything remains coalesced
    for (int i = 0; i < warpSize/numWarps; ++i) {
        int gr = br * warpSize + lane;
        int gc = bc * warpSize + 1 * numWarps + warpId;

        array_out[gc * rows_in + gr] = block[lane][i * numWarps + warpId];
    }
}


template<typename T>
void isTranspose (const std::vector<T> &A, const std::vector<T> &B, int side)
{
    for (int n = 0; n < side; ++n) {
        for (int m = 0; m < side; ++m) {
            assert(A[n * side + m] == B[m * side + n]);
        }
    }
}


int main(void)
{
    const int side = 2048;

    std::vector<int> hIn (side * side);
    std::vector<int> hOut (side * side);

    for (int i = 0; i < side * side; ++i)
      hIn[i] = random() % 100;

    int *dIn, *dOut;
    checkCudaErrors(cudaMalloc(&dIn, sizeof(int) * side * side));
    checkCudaErrors(cudaMalloc(&dOut, sizeof(int) * side * side));
    checkCudaErrors(cudaMemcpy(dIn, &hIn[0], sizeof(int) * side * side, cudaMemcpyHostToDevice));

    const int numThreads = 256;
    const int numBlocks = (side * side + numThreads -1) / numThreads;
    simpleTranspose<<<numBlocks, numThreads>>>(dIn, dOut, side, side);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&hOut[0], dOut, sizeof(int) * side * side, cudaMemcpyDeviceToHost));
    std::cout << "1" << std::endl;
    isTranspose(hIn, hOut, side);

    dim3 bDim(16, 16);
    dim3 gDim(side / 16, side / 16);
    simpleTranspose2D<<<gDim, bDim>>>(dIn, dOut, side, side);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&hOut[0], dOut, sizeof(int) * side * side, cudaMemcpyDeviceToHost));
    std::cout << "2" << std::endl;
    isTranspose(hIn, hOut, side);

    const int warpSize = 32;
    const int numWarpsPerBlock = 4;
    bDim.x = warpSize;
    bDim.y = numWarpsPerBlock;
    gDim.x = side / warpSize;
    gDim.y = side / warpSize;
    fastTranspose<int, numWarpsPerBlock><<<gDim, bDim>>>(dIn, dOut, side, side);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&hOut[0], dOut, sizeof(int) * side * side, cudaMemcpyDeviceToHost));
    std::cout << "3" << std::endl;
    isTranspose(hIn, hOut, side);

    checkCudaErrors(cudaFree(dIn));
    checkCudaErrors(cudaFree(dOut));

    return 0;
    
}
