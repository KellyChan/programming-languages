#include <stdio.h>

#include <cuda_runtime.h>


__global__ void add (int * a, int * b, int * c)
{
  *c = *a + *b;
}


int main (void)
{
  int a, b, c;
  int * d_a, * d_b, * d_c;
  int size = sizeof(int);
   
  // allocate space for device copies of a, b, c
  cudaMalloc ((void**)&d_a, size);
  cudaMalloc ((void**)&d_b, size);
  cudaMalloc ((void**)&d_c, size);

  // setup input values
  a = 2;
  b = 7;
  
  // cuda input to device
  cudaMemcpy (d_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy (d_b, &b, size, cudaMemcpyHostToDevice);

  // launch add() kernel on gpu
  add<<<1, 1>>>(d_a, d_b, d_c);

  // copy result back to host
  cudaMemcpy (&c, d_c, size, cudaMemcpyDeviceToHost);
  printf ("c = a + b = %d + %d = %d\n", a, b, c);

  // cleanup
  cudaFree (d_a);
  cudaFree (d_b);
  cudaFree (d_c);

  return 0;
}
