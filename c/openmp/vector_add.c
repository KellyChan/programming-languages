#include <omp.h>
#define N 10000
#define CHUNKSIZE 100


int main(void)
{
  int i, chunk;
  float a[N], b[N], c[N];

  for (i = 0; i < N; ++i)
  {
    a[i] = b[i] = i * 1.0;
  }
  chunk = CHUNKSIZE;

  #pragma omp parallel shared(a, b, c, chunk) private(i)
  {
    #pragma omp for schedule(dynamic, chunk) nowait
    for (i = 0; i < N; ++i)
      c[i] = a[i] + b[i];
  }

}
