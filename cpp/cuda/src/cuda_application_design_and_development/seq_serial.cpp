#include <vector>
#include <iostream>

using namespace std;


int main()
{
    const int N = 50000;

    // task 1: create the array
    vector<int> a(N);

    // task 2: fill the array
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }

    // task 3: calculate the sum of the array
    int sumA = 0;
    for (int i = 0; i < N; i++)
    {
        sumA += a[i];
    }

    // task 4: calculate the sum of 0..N-1
    int sumCheck = 0;
    for (int i = 0; i < N; i++)
    {
        sumCheck += i;
    }

    // task 5: check the results agree
    if (sumA == sumCheck)
    {
        cout << "Test Succeeded!" << endl;
    } else
    {
        cerr << "Test Failed!" << endl;
        return (1);
    }

    return (0);
}

