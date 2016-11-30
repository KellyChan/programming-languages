
// Device Enumeration
int deviceCount;
CudaGetDeviceCount(&deviceCount);

int device;
for (device = 0; device < deviceCount; ++device)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d\n", device, deviceProp.major, deviceProp.minor);
}


// Device Selection
size_t size = 1024 * sizeof(float);

cudaSetDevice(0);                        // set device 0 as current
float* p0;
cudaMalloc(&p0, size);                   // allocate memory on device 0
MyKernel<<<1000, 128>>>(p0);             // launch kernel on device 0

cudaSetDevice(1);                        // set device 1 as current
float* p1;
cudaMalloc(&p1, size);                   // allocate memory on device 1
MyKernel<<<1000, 128>>>(p1);             // launch kernel ono device 1


// Stream and Event Behavior
cudaSetDevice(0);
cudaStream_t s0;
cudaStreamCreate(&s0);
MyKernel<<<100, 64, 0, s0>>>();          // launch kernel on device 0 in s0

cudaSetDevice(1);
cudaStream_t s1;
cudaStreamCreate(&s1);
MyKernel<<<100, 64, 0, s1>>>();          // launch kernel on device 1 in s1


// Peer-to-Peer Memory Access
size_t size = 1024 * sizeof(float);

cudaSetDevice(0);
float* p0;
cudaMalloc(&p0, size);

cudaSetDevice(1);
float* p1;
cudaMalloc(&p1, size);

cudaSetDevice(0);
MyKernel<<<1000, 128>>>(p0);
cudaSetDevice(1);
cudaMemcpyPeer(p1, 1, p0, 0, size);     // Copy p0 to p1
MyKernel<<<1000, 128>>>(p1);            // launch kernel on device 1
