#include <math.h>

#include "activator.h"


Matrix activator_tanh(const Matrix& X)
{

    int nrows = X.size();
    int ncols = X[0].size();
    Matrix Y_hat(nrows, vector<int>(ncols));

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            Y_hat[i][j] = tanh(X[i][j]);
        }
    }

    return Y_hat;
}
