#include <vector>
#include <iostream>
#include <limits>
#include <cstdlib>

#include "../utils/utils.h"


__device__ int nextPowerOf2 (const int x)
{
    return (1 << (32 - __clz(x - 1)));
}



template<int blockSize>
__global__ void reduceBlockArbitrarySize (const int* const input, int *sum, int N)
{
    const int tid = threadIdx.x;
    __shared__ int smem[blockSize];

    int myVal = 0;
    
    // first do a serial accumulation into each thread's local accumulator
    for (int globalPos = tid; globalPos < N; globalPos += blockSize) {
        myVal += input[globalPos];
    }

    smem[tid] = myVal;
    __syncthreads();

    // once we've reduce the problem to blockSize values, then reduce
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


int main (int argc, char **argv)
{
    int N = atoi(argv[1]);
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

    // const int blockSize = 192;
    const int blockSize = 32;
    reduceBlockArbitrarySize<blockSize><<<1, blockSize>>>(d_input, d_sum, N);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    int h_d_sum;
    checkCudaErrors(cudaMemcpy(&h_d_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "cpu: " << h_sum << " gpu: " << h_d_sum << std::endl;

    return 0;
    
}
