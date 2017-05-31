#include <sys/time.h>
#include <time.h>
#include <stdio.h>


void loop (void)
{
  time_t start, end;
  double elapsed;

  start = time(NULL);
  int counter = 0;
  do {
    counter++;
  } while (counter < 500000000);
  end = time(NULL);
  elapsed = difftime(end, start);
  printf("elapsed: %f", elapsed);
}


void get_clock()
{
    struct timespec tstart={0,0}, tend={0,0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    loop();
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("some_long_computation took about %.5f seconds\n",
           ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - 
           ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
}

int main ()
{
  get_clock();

  return 0;
}
