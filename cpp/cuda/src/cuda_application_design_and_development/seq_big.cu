#include <iostream>

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;


int main()
{
    const int N = 1000000;

    // task 1: create the array
    thrust::device_vector<int> a(N);

    // task 2: fill the array
    thrust::sequence(a.begin(), a.end(), 0);

    // task 3: calculate the sum of the array
    unsigned long long sumA = thrust::reduce(a.begin(),
                                             a.end(),
                                             (unsigned long long) 0,
                                             thrust::plus<unsigned long long>());

    // task 4: calculate the sum of 0..N-1
    unsigned long long sumCheck = 0;
    for (int i = 0; i < N; i++)
    {
        sumCheck += i;
    }

    cerr << "host " << sumCheck << endl;
    cerr << "device " << sumA << endl;

    // task 5: check the results agree
    if (sumA == sumCheck)
    {
        cout << "Test Succeeded!" << endl;
    } else {
        cerr << "Test Failed!" << endl;
        return (1);
    }

    return (0);
}
