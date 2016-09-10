#include "../utils/utils.h"

__global__
void myKernel(int *in) {
    in[threadIdx.x] += 1;
}


int main(void) {
    int *dIn;
    checkCudaErrors(cudaMalloc(&dIn, sizeof(int)));

    myKernel<<<1,2>>>(dIn);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    return 0;
}
