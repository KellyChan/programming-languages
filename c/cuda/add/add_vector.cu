#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// #define N 512
#define N (2048*2048)
#define THREADS_PER_BLOCK 512


__global__ void add (int * a, int * b, int * c)
{
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}


__global__ void add (int * a, int * b, int * c)
{
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}


__global__ void add (int * a, int * b, int * c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}


__global__ void add (int * a, int * b, int * c, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] + b[index];
}


void random_ints (int * a, int count)
{
  for (int i = 0; i < count; ++i)
    a[i] = (int)rand();
}


int main (void)
{
  int * a, * b, * c;
  int * d_a, * d_b, * d_c;
  int size = N * sizeof(int);
   
  // allocate space for device copies of a, b, c
  cudaMalloc ((void**)&d_a, size);
  cudaMalloc ((void**)&d_b, size);
  cudaMalloc ((void**)&d_c, size);

  // setup input values
  a = (int *)malloc(size); random_ints (a, N);
  b = (int *)malloc(size); random_ints (b, N);
  c = (int *)malloc(size); 
 
  // cuda input to device
  cudaMemcpy (d_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_b, &b, size, cudaMemcpyHostToDevice);

  // launch add() kernel on gpu
  // add<<<N, 1>>>(d_a, d_b, d_c);
  // add<<<1, N>>>(d_a, d_b, d_c);
  // add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
  add<<<(N+M-1)/M, M>>>(d_a, d_b, d_c, N);

  // copy result back to host
  cudaMemcpy (&c, d_c, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i)
    printf ("c[%d] = a[%d] + b[%d] = %d + %d = %d\n", i, i, i, a[i], b[i], c[i]);

  // cleanup
  cudaFree (d_a);
  cudaFree (d_b);
  cudaFree (d_c);
  free (a);
  free (b);
  free (c);

  return 0;
}
