#include <stdio.h>


__global__ void childKernel()
{
    printf("Hello ");
}


__global__ void parentKernel()
{
    // launch child
    childKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError())
    {
        return;
    }

    // wait for child to complete
    if (cudaSuccess != cudaDeviceSynchronize())
    {
        return;
    }

    printf("World!\n");
}


int main(int argc, char* argv[])
{
    // launch parent
    parentKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError())
    {
        return 1;
    }

    // wait for parent to complete
    if (cudaSuccess != cudaDeviceSynchronize())
    {
        return 2;
    }

    return 0;
}
