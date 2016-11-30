// Global variables
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));


__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));


__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));


// Device Host: kernel definition
__global__ void 2DKernel(float* devPtr, size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r)
    {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c)
        {
            float element = row[c];
        }
    }
} 


__global__ void 3DKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z)
    {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y)
        {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++i)
            {
                float element = row[x];
            }
        }
    }
}


// Host code
int main()
{

    // Allocate memory with width x height
    int width = 64;
    int height = 64;
    float* devPtr;
    size_t pitch;
    cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
    2DKernel<<<100, 512>>>(devPtr, pitch, width, height);

    // Allocate memory with width x height x depth
    int width = 64;
    int height = 64;
    int depth = 64;
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth));
    cudaPitchedPtr devPitchedPtr;
    cudaMalloc3D(&devPitchedPtr, extent);
    3DKernel<<<100, 512>>>(devPicthedPtr, width, height, depth);


}
