// Device Host: kernel definition
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    // int i = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}



__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}


__global__ void MatAddPlus(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j]
}



// Host code
int main()
{

    int N = 1;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    //

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    // Free host memory

    // Kernel invocation with one block of N * N * 1 threads
    // int numBlocks = 1;
    // dim3 threadsPerBlock(N, N);
    // MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

    // Kernel invocation: block -> threads
    // dim3 threadsPerBlock(16, 16);  // a thread block size: 16 * 16 = 256 threads
    // dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // MatAddPlus<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
