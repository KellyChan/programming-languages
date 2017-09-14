#include <omp.h>
#include <stdio.h>


int main (void)
{
  int nthreads, tid;

  #pragma omp parallel private(tid)
  {
    tid = omp_get_thread_num();
    printf("Hello world from thread = %d\n", tid);

    if (tid == 0)
    {
      nthreads = omp_get_num_threads();
      printf("number of threads = %d\n", nthreads);
    }
  }
}
