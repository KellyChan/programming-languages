#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <vector>

using namespace std;

typedef vector< vector<int> > Matrix;

Matrix weights_generator(int rows, int cols);
Matrix zero_weights_generator(int rows, int cols);

#endif
