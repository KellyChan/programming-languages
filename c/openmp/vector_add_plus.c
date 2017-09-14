#include <omp.h>
#define N 1000


int main(void)
{
  int i;
  float a[N], b[N], c[N], d[N];

  for (i = 0; i < N; ++i)
  {
    a[i] = i * 1.5;
    b[i] = i + 22.35;
  }

  #pragma omp parallel shared(a, b, c, d) private(i)
  {
    #pragma omp sections nowait
    {
      #pragma omp section
      for (i = 0; i < N; ++i)
        c[i] = a[i] + b[i];

      #pragma omp section
      for (i = 0; i < N; ++i)
        d[i] = a[i] * b[i];
    }
  }
}
