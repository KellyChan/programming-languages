#include <tuple>
#include <iostream>
#include <complex>
#include <string>

using namespace std;


int main()
{
    tuple<string, int, int, complex<double>> t;
    tuple<int, float, string> t1(41, 6.2, "nico");

    cout << get<0>(t1) << " ";
    cout << get<1>(t1) << " ";
    cout << get<2>(t1) << " ";
    cout << endl;

    auto t2 = make_tuple(22, 44, "nico");
    get<1>(t1) = get<1>(t2);

    if (t1 < t2) {
        t1 = t2;
    }
}
