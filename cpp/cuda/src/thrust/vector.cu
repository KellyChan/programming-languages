#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;


int main(void)
{

    thrust::host_vector<float> H(4000);
    cout << "H size: " << H.size() << endl;    
    // initialize
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;
   
    // H.resize(2); 
    cout << "H size: " << H.size() << endl;    

    thrust::device_vector<float> D = H;
    D[0] = 99;
    D[1] = 88;
    cout << "D size: " << D.size() << endl;

    return 0;
}
