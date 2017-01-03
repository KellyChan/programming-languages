#include <vector>

#include "linear.h"

using namespace std;


Matrix linear_fn_without_bias(const Matrix& W, const Matrix& X)
{
    int n = W.size();
    int m = W[0].size();
    int p = X[0].size();
    Matrix Y_Hat(n, vector<int>(p, 0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                Y_Hat[i][j] += W[i][k] * X[k][j];
            }
        }
    }

    return Y_Hat;
}
