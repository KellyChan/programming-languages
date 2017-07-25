#include <iostream>


int main ()
{
  #pragma omp parallel
  {
    std::cout << "All threads run this\n";
    #pragma omp sections
    {
      #pragma omp section
      {
        std::cout << "This executes in parallel\n";
      }
      #pragma omp section
      {
        std::cout << "Sequential statement\n";
        std::cout << "This always executes after statement\n";
      }
      #pragma omp section
      {
        std::cout << "This also executes in parallel.\n";
      }
    }
  }
}
