#include <stdio.h>

__global__ void print (size_t n)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    printf("index: %d, blockIdx: %d, threadIdx: %d, blockDim: %d, gridDim: %d\n", i, blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
  }
}


int main(void)
{
  print<<<1,1024>>>(2048);
  cudaDeviceSynchronize();
}
