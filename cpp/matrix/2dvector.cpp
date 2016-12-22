#include <vector>
#include <iostream>

using namespace std;


int main()
{
    vector < vector<int> > Matrix(5, vector<int>(4, 1));

    for (auto vec: Matrix)
    {
        for (auto x: vec)
        {
            cout << x << ",";
        }
    }


    return 0;
}
