#include <iostream>
#include <limits>
#include <string>

using namespace std;


int main()
{
    cout << boolalpha;

    // integer
    cout << "max(short): " << numeric_limits<short>::max() << endl;
    cout << "max(int): " << numeric_limits<int>::max() << endl;
    cout << "max(long): " << numeric_limits<long>::max() << endl;
    cout << endl;

    // float
    cout << "max(float): " << numeric_limits<float>::max() << endl;
    cout << "max(double): " << numeric_limits<double>::max() << endl;
    cout << "max(long double): " << numeric_limits<long double>::max() << endl;
    cout << endl;

    // char
    cout << "is_signed(char): " << numeric_limits<char>::is_signed << endl;
    cout << endl;

    // string
    cout << "is_specifialized(string): " << numeric_limits<string>::is_specialized << endl;
}
