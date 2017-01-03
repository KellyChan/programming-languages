#include <vector>

#include "matrix.h"

using namespace std;

/*

One Matrix

*/

void swap(int& a, int& b)
{
    int c = a;
    a = b;
    b = c;
}

Matrix matrix_transpose(const Matrix& X)
{
    int nrows = X.size();
    int ncols = X[0].size();
    Matrix X_T(ncols, vector<int>(nrows));

    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            X_T[i][j] = X[j][i];
        }
    }

    return X_T;
}



// Is a matrix symmetric
bool matrix_is_symmetric(const Matrix& X)
{
    int n = X.size();
    for (int i = 0; i < n-1; ++i)
    {
        for (int j = i+1; j < n; ++j)
        {
            if (X[i][j] != X[j][i])
                return false;
        }
    }
    return true;
}


bool matrix_search(const Matrix& X, int element)
{
    int nrows = X.size();
    int ncols = X[0].size();

    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            if (X[i][j] == element)
                return true;
        }
    }
    return false;
}


Matrix matrix_product(const Matrix& X, float lambda)
{
    int nrows = X.size();
    int ncols = X[0].size();
    Matrix Y(nrows, vector<int>(ncols));

    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            Y[i][j] = lambda * X[i][j];
        }
    }
    return Y;
}


Matrix matrix_divide(const Matrix& X, float lambda)
{
    int nrows = X.size();
    int ncols = X[0].size();
    Matrix Y(nrows, vector<int>(ncols));

    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            Y[i][j] = X[i][j] / lambda;
        }
    }
    return Y;
}


/*

Two Matrices

*/

Matrix matrix_sum(const Matrix& X1, const Matrix& X2)
{
    int nrows = X1.size();
    int ncols = X1[0].size();
    Matrix Y(nrows, vector<int>(ncols));

    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            Y[i][j] = X1[i][j] + X2[i][j];
        }
    }

    return Y;
}


// Matrix multiplcation
Matrix matrix_multiply(const Matrix& X1, const Matrix& X2)
{
    int n = X1.size();
    int m = X1[0].size();
    int p = X2[0].size();
    Matrix Y(n, vector<int>(p, 0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                Y[i][j] += X1[i][k] * X2[k][j];
            }
        }
    }

    return Y;
}
