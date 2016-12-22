#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <limits>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>


int main(void) 
{

    const int N = 100000000;
    std::vector<float> numbers(N);

    for (int i = 0; i < N; ++i)
      numbers[i] = (rand() % 1000) / 1000.;
    thrust::device_vector<float> d_numbers = numbers;
    float thrust_sum = thrust::reduce(d_numbers.begin(), d_numbers.end());
    std::cout << "sum gpu tree reduction: " << std::setprecision(20) << thrust_sum << std::endl;

    float sum = 0.f;
    for (int i = 0; i < N; ++i)
      sum += numbers[i];
    std::cout << "sum float: " << std::setprecision(20) << sum << std::endl;

    double sum_d = 0;
    for (int i = 0; i < N; ++i)
      sum_d += numbers[i];
    std::cout << "sum double: " << std::setprecision(20) << sum_d << std::endl;

    std::sort(numbers.begin(), numbers.end());
    std::cout << "\nAfter sorting\n";

    sum = 0.f;
    for (int i = 0; i < N; ++i)
      sum += numbers[i];
    std::cout << "sum float: " << std::setprecision(20) << sum << std::endl;

    sum_d = 0.;
    for (int i = 0; i < N; ++i)
      sum_d += numbers[i];
    std::cout << "sum double: " << std::setprecision(20) << sum_d << std::endl;

    return 0;

}
