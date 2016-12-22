#include <time.h>
#include <string.h>
#include <stdlib.h>

// for printing
#include <iomanip>
#include <iostream>

// eigen
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>


using namespace std;
using namespace Eigen;


int main(int argc, char** argv)
{
    int M = 3;
    int N = 5;
    MatrixXd X(M, N);  // define an MxN general matrix

    // Fill X by random numbers between 0 and 9
    // Note that indexing into matrices in NewMat is 1-based
    srand(time(NULL));
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            X(i, j) = rand() % 10;
        }
    }

    MatrixXd C;
    C = X * X.transpose();  // C = X * X_T
    cout << "The symmetrix matrix C" << endl;
    cout << C << endl;

    // compute eigendecomposition of C
    SelfAdjointEigenSolver<MatrixXd> es(C);
    MatrixXd D = es.eigenvalues().asDiagonal();
    MatrixXd V = es.eigenvectors();

    cout << "The eigenvalues matrix: " << endl;
    cout << D << endl;
    cout << "The eigenvectors matrix: " << endl;
    cout << V << endl;

    // Check that the first eigenvector indeed has the eigenvector property
    VectorXd v1(3);
    v1(0) = V(0, 0);
    v1(1) = V(1, 0);
    v1(2) = V(2, 0);

    VectorXd Cv1 = C * v1;
    VectorXd lambda1_v1 = D(0) * v1;

    cout << "The max-norm of the difference between C*v1 and lambda*v1 is " << endl;
    cout << Cv1.cwiseMax(lambda1_v1) << endl << endl;

    // Build the inverse and check the result
    MatrixXd Ci = C.inverse();
    MatrixXd I = Ci * C;

    cout << "The inverse of C is " << endl;
    cout << Ci << endl;
    cout << "And the inverse times C is identity" << endl;
    cout << I << endl;

    // Example of multiple solves
    VectorXd r1(3);
    VectorXd r2(3);
    for (int i = 0; i < 3; ++i)
    {
        r1(i) = rand() % 10;
        r2(i) = rand() % 10;
    }

    ColPivHouseholderQR<MatrixXd> qr(C);  // decomposes C
    VectorXd s1 = qr.solve(r1);
    VectorXd s2 = qr.solve(r2);

    cout << "Solution for right hand side r1: " << endl;
    cout << s1 << endl;
    cout << "Solution for right hand side r1: " << endl;
    cout << s2 << endl;

    return 0;
}
