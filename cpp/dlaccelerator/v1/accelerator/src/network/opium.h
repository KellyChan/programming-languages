#ifndef OPIUM_H
#define OPIUM_H

#include <vector>

#include "../utils/matrix.h"

using namespace std;

typedef vector< vector<int> > Matrix;

Matrix opium_lite(const Matrix& hidden_output,
                  const Matrix& loss,
                  const Matrix& W2,
                  float theta);


#endif
