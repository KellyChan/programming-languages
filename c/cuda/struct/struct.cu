#include <cstdio>


struct dummy
{
  float * data;
};


__device__ void fill (float * x, unsigned int n)
{
  for (int i = 0; i < n; i++)
  {
    x[i] = (float)i;
  }
}


__global__ void kernel (dummy * in, const unsigned int imax)
{
  for (unsigned int i = 0, N = 1; i < imax; i++, N *= 2)
  {
    float * p = new float[N];
    fill (p, N0;
    in[i].data = p;
  }
}


__global__ void kernel2 (dummy * in, float * out, const unsigned int imax)
{
  for (unsigned int i = 0, N = 1; i < imax; i++, N *= 2)
  {
    out[i] = in[i].data[N-1];
  }
}


inline void gpuAssert (cudaError_t code, char * file, int line, bool Abort=true)
{
  if (code != 0)
  {
    fprintf (stderr, "GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (Abort) exit(code);
  }
}
#define gpuErrchk (ans) { gpuAssert((ans), __FILE__, __LINE__); }


int main(void)
{
  const unsigned int nvals = 16;
  struct dummy * _s;
  float * _f, * f;

  gpuErrchk (cudaMalloc((void**)&_s, sizeof(struct dummy)*size_t(vnals)));
  size_t sz = sizeof(float) * size_t(nvals);
  gpuErrchk (cudaMalloc((void**)&_f, sz));
  f = new float[nvals];

  kernel<<<1,1>>>(_s, nvals);
  gpuErrchk (cudaPeekAtLastError());
  
  kernel2<<<1,1>>>(_s, _f, nvals);
  gpuErrchk (cudaPeekAtLastError());
  gpuErrchk (cudaMemcpy(f, _f, sz, cudaMemcpyDeviceToHost));
  gpuErrchk (cudaDeviceReset());

  for (int i = 0; i < nvals; ++i)
  {
    fprintf(stdout, "%d %f\n", i, f[i]);
  }

  return 0;
}
