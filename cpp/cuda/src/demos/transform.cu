#include <iostream>
#include <algorithm>
#include <vector>

#include "../utils/utils.h"
#include "../utils/timer.h"


template<typename T, typename F>
__global__ void deviceTransform(T *input, T *output, int N, F op) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i<N; i+=gridDim.x * blockDim.x) {
        output[i] = op(input[i]);
    }
}


template<typename T>
struct square{
    __host__ __device__
    T operator()(const T &x) {
        return x * x;
    }
};


int main(void) {

    std::vector<int> input(100000000);
    std::vector<int> output(input.size());
    std::vector<int> device_output(input.size());

    for (int i=0; i < input.size(); ++i)
      input[i] = rand() % 100;

    int *d_input, *d_output;
    checkCudaErrors(cudaMalloc(&d_input, sizeof(int) * input.size()));
    checkCudaErrors(cudaMalloc(&d_output, sizeof(int) * input.size()));
    checkCudaErrors(cudaMemcpy(d_input, &input[0], sizeof(int) * input.size(), cudaMemcpyHostToDevice));

    GpuTimer timer; timer.Start();
    std::transform(input.begin(), input.end(), output.begin(), square<int>());
    timer.Stop(); std::cout << "CPU took: " << timer.Elapsed() << " ms" << std::endl;

    const int blockSize = 192;
    const int gridSize = std::min( ((int)input.size() + blockSize -1)/blockSize, 65535 );

    timer.Start();
    deviceTransform<<<gridSize, blockSize>>>(d_input, d_output, input.size(), square<int>());
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    timer.Stop(); std::cout << "GPU took: " << timer.Elapsed() << " ms" << std::endl;

    checkCudaErrors(cudaMemcpy(&device_output[0], d_output, sizeof(int) * input.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < input.size(); ++i) {
        if ( i<10 )
          std::cout << input[i] << " " << output[i] << " " << device_output[i] << std::endl;
        
        if ( device_output[i] != output[i] ) {
          std::cerr << "CPU and GPU results don't match at " << i << std::endl;
        }
    }

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}
