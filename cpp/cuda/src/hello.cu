#include <iostream>
#include <vector>
#include <cstdlib>

#include "utils/utils.h"


__global__
void kernel(int *out) {
    out[threadIdx.x] = threadIdx.x;
}


int main(int argc, char **argv) {

    int N = 32;

    if (argc == 2)
      N = atoi(argv[1]);

    int *d_output;
    std::vector<int> h_output(N);
    checkCudaErrors(cudaMalloc(&d_output, sizeof(int) * N));

    kernel<<<1, N>>>(d_output);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&h_output[0], d_output, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i=0; i<N; ++i) {
        std::cout << h_output[i] << std::endl;
    }
    checkCudaErrors(cudaFree(d_output));

    return 0;
}
