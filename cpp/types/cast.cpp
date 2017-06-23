#include <iostream>

int main ()
{
  int a = -1;
  int b = 500;
  std::cout << a << "," << (unsigned)(a) << std::endl;
  std::cout << a << "," << static_cast<unsigned>(a) << std::endl;
  std::cout << b << "," << static_cast<unsigned>(b) << std::endl;

  return 0;
}
