#include <vector>
#include <iostream>
#include <limits>

#include "../utils/utils.h"


__global__ void reduceKernel (const float* const input, float *sum, int N)
{
    const int lane = threadIdx.x;
    // const int warp = threadIdx.y;

    __shared__ float smem[32];
    float myVal = input[lane];
    smem[lane] = myVal;

    __syncthreads();

    for (int shift = 16; shift > 0; shift >>= 1) {
        if (lane < shift) {
            smem[lane] += smem[lane + shift];
        }
        __syncthreads();
    }

    if (lane == 0)
      *sum = smem[lane];

}


int main(void) 
{
    const int N = 32;
    std::vector<float> h_input(N);

    float h_sum = 0.f;
    for (int i = 0; i < N; ++i) {
        h_input[i] = (rand() / (double)std::numeric_limits<int>::max());
        h_sum += h_input[i];
    }

    float *d_input;
    checkCudaErrors(cudaMalloc(&d_input, N * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_input, &h_input[0], N * sizeof(int), cudaMemcpyHostToDevice));

    float *d_sum;
    checkCudaErrors(cudaMalloc(&d_sum, sizeof(int)));

    reduceKernel<<<1, 32>>>(d_input, d_sum, N);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    float h_d_sum;
    checkCudaErrors(cudaMemcpy(&h_d_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "cpu: " << h_sum << " gpu: " << h_d_sum << std::endl;

    return 0;

}
