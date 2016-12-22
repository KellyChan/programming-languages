#include <iostream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;


int main()
{
    const int nrows = 4;
    const int ncols = 5;

    MatrixXd X(nrows, ncols);
    X.setRandom();
    cout << "-- X -- " << endl << X << endl;

    MatrixXd Y(ncols, nrows);
    Y.setRandom();
    cout << "-- Y -- " << endl << Y << endl;

    MatrixXd Z = X.matrix() * Y.matrix();
    cout << "-- Z -- " << endl << Z << endl;

    return 0;
}
