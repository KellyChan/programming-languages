#ifndef ELM_H
#define ELM_H

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Cholesky>

using namespace std;
using namespace Eigen;


int compare(const void* a, const void* b);

// template <typename Derived>
MatrixXd buildTargetMatrix(double* Y, int nlabels);

// entry function to train the ELM model
// Input: X, Y, nhn, C
// Output: inW, bias, outW
template <typename Derived>
int elmTrain(double* X, int dims, int nsmp,
             double* Y,
             const int nhn, const double C,
             MatrixBase<Derived>& intW,
             MatrixBase<Derived>& bias,
             MatrixBase<Derived>& outW)
{
    // map the samples into the matrix object
    MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);

    // build target matrix
    MatrixXd mTargets = buildTargetMatrix(Y, nsmp);

    // generate random input weight matrix - inW
    inW = MatrixXd::Random(nhn, dims);

    // generate random bias vectors
    bias = MatrixXd::Random(nhn, 1);

    // compute the pre-H matrix
    MatrixXd preH = inW * mX + bias.replicate(1, nsmp);

    // compute hidden neuron output
    MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();

    // build matrices to solve Ax = b
    MatrixXd A = (MatrixXd::Identity(nhn, nhn)).array() * (1/C) + (H * H.transpose()).array();
    MatrixXd b = H * mTargets.transponse();

    // solve the output weights as a solution to a system of linear equations
    outW = A.llt().solve(b);

    return 0;
}


// entry function to predict class labels using the trained ELM model on test data
// Input: X, inW, bias, outW
// Output: scores
template <typename Derived>
int elmPredict(double* X, int dims, int nsmp,
               MatrixBase<Derived>& mScores,
               MatrixBase<Derived>& intW,
               MatrixBase<Derived>& bias,
               MatrixBase<Derived>& outW)
{
    // map the sample into the Eigen's matrix object
    MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);

    // build the pre-H matrix
    MatrixXd preH = inW * mX + bias.replicate(1, nsmp);

    // apply the activation function
    MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();

    // compute output scores
    mScores = (H.transpose() * outW).transpose();

    return 0;
}

// ---------------------------------------------------------------
// implementations

int compare(const void* a, const void* b)
{
    const double* da = (const double*) a;
    const double* db = (const double*) b;
    return (*da > *db) - (*da < *db);
}


// builds 1-of-K target matrix from labels array
// template <typename Derived>
MatrixXd buildTargetMatrix(double* Y, int nlabels)
{
    // make a temporary copy of the labels array
    double* tmpY = new double[nlabels];
    for (int i = 0; i < nlabels; i++)
    {
        tmpY[i] = Y[i];
    }

    // sort the array of labels
    qsort(tmpY, nlabels, sizeof(double), compare);

    // count unique labels
    int nunique = 1;
    for (int i = 0; i < nlabels; i++)
    {
        if (tmpY[i] != tmpY[i+1])
            nunique++;
    }

    delete [] tmpY;

    MatrixXd targets(nuique, nlabels);
    targets.fill(0);

    // fill in the ones
    for (int i = 0; i < nlabels; i++)
    {
        int idx = Y[i] - 1;
        targets(idx, i) = 1;
    }

    // normalize the targets matrix values (-1/1)
    targets *= 2;
    targets.array() -= 1;

    return targets;
}

#endif
