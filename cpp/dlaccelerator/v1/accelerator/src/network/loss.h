#ifndef LOSS_H
#define LOSS_H

#include <vector>

using namespace std;

typedef vector< vector<int> > Matrix;

Matrix deviation(const Matrix& Y, const Matrix& Y_hat);


#endif
