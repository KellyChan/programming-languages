#ifndef LINEAR_H
#define LINEAR_H

#include <vector>

using namespace std;


typedef vector< vector<int> > Matrix;


Matrix linear_fn_without_bias(const Matrix& W, const Matrix& X);


#endif
