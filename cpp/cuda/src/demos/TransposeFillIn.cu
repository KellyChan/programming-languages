#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <vector>

#include "../utils/utils.h"


template<typename T, int numWarps>
__global__ void fastTranspose(const T *array_in, const T * array_out, int rows_in, int cols_in)
{
    const int warpId = threadIdx.y;
    const int lane = threadIdx.x;
    const int warpSize = 32;
}


template<typename T>
void isTranspose(const std::vector<T> &A, const std::vector<T> &B, int side)
{
    for (int n = 0; n < side; ++n) {
        for (int m = 0; m < side; ++m) {
            assert(A[n * side + m] == B[m * side + n]);
        }
    }
}


int main(void) {

    const int side = 2048;

    std::vector<int> hIn (side * side);
    std::vector<int> hOut (side * side);

    for (int i = 0; i < side * side; ++i) {
        hIn[i] = random() % 100;
    }

    int *dIn, *dOut;
    checkCudaErrors(cudaMalloc(&dIn, sizeof(int) * side * side));
    checkCudaErrors(cudaMalloc(&dOut, sizeof(int) * side * side));
    checkCudaErrors(cudaMemcpy(dIn, &hIn[0], sizeof(int) * side * side, cudaMemcpyHostToDevice));

    const int warpSize = 32;
    const int numWarpsPerBlock = 4;
    dim3 bDim, gDim;

    bDim.x = warpSize;
    bDim.y = numWarpsPerBlock;
    gDim.x = ;
    gDim.y = ;

    fastTranspose<int, numWarpsPerBlock><<<gDim, bDim>>>(dIn, dOut, side, side);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&hOut[0], dOut, sizeof(int) * side * side, cudaMemcpyDeviceToHost));
    isTranspose(hIn, hOut, side);
 
    checkCudaErrors(cudaFree(dIn));
    checkCudaErrors(cudaFree(dOut));

    return 0;

}
