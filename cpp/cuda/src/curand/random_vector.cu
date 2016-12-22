#include <cuda.h>
#include <curand.h> 


__global__ void init_stuff(curandState *state) {

 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 curand_init(1337, idx, 0, &state[idx]);

}

__global__ void make_rand(curandState *state, float

*randArray) {

 int idx = blockIdx.x * blockDim.x + threadIdx.x;

 randArray[idx] = curand_uniform(&state[idx]);

}

void host_function() {

 curandState *d_state;

 cudaMalloc(&d_state, nThreads * nBlocks);

 init_stuff<<<nblocks, nthreads>>>(d_state);

 make_rand<<<nblocks, nthreads>>>(d_state, randArray);

 cudaFree(d_state);

}

host_function();
