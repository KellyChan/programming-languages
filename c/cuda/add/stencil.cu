__global__ void stencil_1d (int * in, int * out)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockdim.x;
  int lindex = threadIdx.x + RADIUS;

  // read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS)
  {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // synchronize (ensure all the data is available)
  __syncthreads();

  // apply the sstencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += temp[lindex + offset];

  // store the result
  out[gindex] = result;
}
