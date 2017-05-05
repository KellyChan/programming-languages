#include <iostream>


int main()
{
  
  int N = 1;
  int C = 3;
  int H = 8;
  int W = 8;
  int outer_num_ = N * C;
  int inner_num_ = H * W;

  int dim = N * C * H * W  / outer_num_;

  for (int i = 0; i < outer_num_; ++i)
  {
    for (int j = 0; j < C; j++)
    {
      std::cout << "\nc = " << j << ", ";
      for (int k = 0; k < inner_num_; ++k)
      { 
         std::cout << "\nk = " << k << ", ";
         std::cout << "bottom = " << i * dim + j * inner_num_ + k << ", "; 
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
