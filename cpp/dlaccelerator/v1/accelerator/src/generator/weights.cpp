#include <random>
#include <iterator>
#include <algorithm>
#include <functional>

#include <iostream>

#include "weights.h"

using namespace std;


Matrix weights_generator(int rows, int cols)
{
    Matrix this_matrix(rows, vector<int>(cols));

    random_device rnd_device;
    mt19937 mersenne_engine(rnd_device());
    uniform_int_distribution<int> dist(-1, 1);
    auto gen = bind(dist, mersenne_engine);

    for (int i = 0; i < this_matrix.size(); i++)
    {
        generate(begin(this_matrix[i]), end(this_matrix[i]), gen);
    }

   
    /* 
    for (int i = 0; i < this_matrix.size(); i++)
    {
        for (int j = 0; j < this_matrix[i].size(); j++)
        {
            cout << this_matrix[i][j] << " ";
        }
    }
    */

    return this_matrix;
}


Matrix zero_weights_generator(int rows, int cols)
{
    Matrix this_matrix(rows, vector<int>(cols, 0));

    return this_matrix;
}
