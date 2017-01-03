#include "loss.h"


Matrix deviation(const Matrix& Y, const Matrix& Y_hat)
{

    int nrows = Y.size();
    int ncols = Y[0].size();
    Matrix loss(nrows, vector<int>(ncols));

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            loss[i][j] = Y[i][j] - Y_hat[i][j];
        }
    }

    return loss;

}
