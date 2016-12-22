#include<iostream>

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;


int main()
{
    const int N = 50000;

    // task 1: create the array
    thrust::device_vector<int> a(N);

    // task 2: fill the array
    thrust::sequence(a.begin(), a.end(), 0);

    // task 4: calculate the sum of 0..N-1
    int sumCheck = 0;
    #pragma omp parallel for reduction(+: sumCheck)
    for (int i = 0; i < N; i++)
    {
        sumCheck += i;
    }

    // task 3: calculate the sum of the array
    int sumA = thrust::reduce(a.begin(), a.end(), 0);

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
