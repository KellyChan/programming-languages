#include <iostream>


int main ()
{
  #pragma omp parallel num_threads (5)
  {
    std::cout << "Hello World!\n";
  }
}
