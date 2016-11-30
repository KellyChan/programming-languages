#include <stdio.h>

# define N 10


__global__ void AddKernel(int a, int b, int* c)
{
    *c = a + b;
}


__global__ void AddPlusKernel(int* a, int* b, int* c)
{
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}



int add(void)
{
    int c;
    int* d_c;
    cudaMalloc((void**)&d_c, sizeof(int));
    AddKernel<<<1,1>>>(2, 7, d_c);
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2 + 7 = %d\n", c);

    cudaFree(d_c);

    return 0;
}


int addgpu(void)
{

    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    // Allocate the memory on the GPU
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMalloc((void**)&d_b, N*sizeof(int));
    cudaMalloc((void**)&d_c, N*sizeof(int));

    // Fill the arrays 'a' and 'b' on the GPU
    for (int i=0; i<N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // Copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

    AddPlusKernel<<<N, 1>>>(d_a, d_b, d_c);

    // Copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Display the results
    printf(" --- GPU --- \n");
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free the memory allocated on the GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;    
}


int main(void)
{
    add();
    addgpu();

    return 0;
}
