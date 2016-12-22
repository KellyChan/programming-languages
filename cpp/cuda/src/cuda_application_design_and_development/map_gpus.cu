#include <iostream>

#include <thrust/sequence.h>
#include <thrust/device_vector.h>

using namespace std;


int main(int argc, char* argv[])
{
    int nGPU;
    if (argc < 2)
    {
        cerr << "Use: number of integers" << endl;
        return (1);
    }

    cudaGetDeviceCount(&nGPU);

    int n = atoi(argv[1]);
    int size = nGPU * n * sizeof(int);

    cout << "nGPU" << nGPU << "" << (n * nGPU * sizeof(int) / 1e6) << "MB" << endl;

    int *h_A;
    cudaHostAlloc(&h_A, size, cudaHostAllocMapped);

    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(i);
        thrust::sequence(thrust::device_pointer_cast(h_A + i * n),
                         thrust::device_pointer_cast(h_A + (i + 1) * n),
                         i * n);
    }
    cudaDeviceSynchronize();  // synchronize the writes

    for (int i = 0; i < nGPU*n; i++)
    {
        if (h_A[i] != i)
        {
            cout << "Error " << h_A[i] << endl;
            exit(1);
        }
    }

    cout << "Success!\n" << endl;
    cudaFreeHost(h_A);

    return (0);
}
