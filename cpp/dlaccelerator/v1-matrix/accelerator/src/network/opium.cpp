#include "opium.h"

// OPIUM Lite
//
//    norm    = (1.0 / theta) + (Activation.T * Activation)
//    weights = weights + (error * Activation.T) / norm
Matrix opium_lite(const Matrix& hidden_output,
                  const Matrix& loss,
                  const Matrix& W2,
                  float theta)
{
    // Calculate norm
    Matrix W2_T = matrix_transpose(W2);
    Matrix W2_dot = matrix_multiply(W2_T, W2);   

    Matrix bias(1, vector<int>(1));
    bias[0][0] = (float) 1.0 / theta;
    
    float norm;
    norm = bias[0][0] + W2_dot[0][0];

    // Calculate weights
    Matrix Loss_dot = matrix_multiply(loss, W2_T);
    int nrows = Loss_dot.size();
    int ncols = Loss_dot[0].size();
    Matrix step_towards(nrows, vector<int>(ncols));
    step_towards = matrix_divide(Loss_dot, norm);

    Matrix this_W2(W2.size(), vector<int>(W2[0].size()));
    this_W2 = matrix_sum(W2, step_towards);

    return this_W2;
}
