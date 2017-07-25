#include <omp.h>
#include <iostream>


int main ()
{
  omp_set_num_threads (5);
  #pragma omp parallel
  {
    std::cout << "Hello world!\n";
  }
}
