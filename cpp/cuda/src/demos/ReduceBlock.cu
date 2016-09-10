#include <vector>
#include <iostream>
#include <limits>

#include "../utils/utils.h"


__device__ int nextPowerOf2 (const int x)
{
    return (1 << (32 - __clz(x - 1)));
}


template<int blockSize>
__global__ void reduceBlock (const int* const input, int *sum)
{
    const int tid = threadIdx.x;
    __shared__ int smem[blockSize];
    int myVal = input[tid];
    smem[tid] = myVal;
    __syncthreads();

    // use this for non-power of 2 blockSizes
    for (int shift = nextPowerOf2(blockSize) / 2; shift > 0; shift >>= 1) {
        if (tid < shift && tid + shift < blockSize) {
            smem[tid] += smem[tid + shift];
        }
        __syncthreads();
    }

    if (tid == 0)
      *sum = smem[tid];
}


int main(void)
{
    const int N = 121;
    std::vector<int> h_input(N);
    
    int h_sum = 0.f;
    for (int i = 0; i < N; ++i) {
        h_input[i] = rand() % 10;
        h_sum += h_input[i];
    }

    int *d_input;
    checkCudaErrors(cudaMalloc(&d_input, N * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_input, &h_input[0], N * sizeof(int), cudaMemcpyHostToDevice));

    int *d_sum;
    checkCudaErrors(cudaMalloc(&d_sum, sizeof(int)));

    reduceBlock<N><<<1, N>>>(d_input, d_sum);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    int h_d_sum;
    checkCudaErrors(cudaMemcpy(&h_d_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "cpu: " << h_sum << " gpu: " << h_d_sum << std::endl;

    return 0;
}
