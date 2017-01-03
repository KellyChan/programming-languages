#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

using namespace std;

typedef vector< vector<int> > Matrix;


/*

One Matrix

*/

void swap(int& a, int&b);
Matrix matrix_transpose(const Matrix& X);

bool matrix_is_symmetric(const Matrix& X);
bool matrix_search(const Matrix& X, int element);

Matrix matrix_product(Matrix& X, float lambda);
Matrix matrix_divide(const Matrix& X, float lambda);

/*

Two Matrices

*/

Matrix matrix_sum(const Matrix& X1, const Matrix& Y);
Matrix matrix_multiply(const Matrix& X1, const Matrix& X2);


#endif
