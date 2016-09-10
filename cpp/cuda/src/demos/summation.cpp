#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <limits>


int main(void) {

    const int N = 100000000;

    std::vector<float> numbers(N);

    for (int i = 0; i < N; ++i)
      numbers[i] = ((double)rand() / (double)std::numeric_limits<int>::max());

    float sum = 0.f;
    for (int i = 0; i < N; ++i)
      sum += numbers[i];
    std::cout << "sum float: " << std::setprecision(20) << sum << std::endl;

    double sum_d = 0.;
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
