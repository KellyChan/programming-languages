#include <iostream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;


int main()
{
    // user data
    double data[6] = { 1, 2, 3, 4, 5, 6 };

    // creating a 3x2 matrix from user data
    MatrixXd mat = Map<MatrixXd>( data, 3, 2 );

    // output it to see what is inside
    cout << mat << endl;

    return 0;
}
