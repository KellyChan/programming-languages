#include <iostream>
#include <iomanip>
#include <limits>
#include <xmmintrin.h>


void enableFtzDas()
{
    int mxcsr = _mm_getcsr();
    mxcsr |= ( 1 << 15 ) | ( 1 << 11 );
    mxcsr |= ( 1 << 6 );

    _mm_setcsr(mxcsr);
}


int main(void)
{
    double a = .1;
    double b = .2;
    double c = .3;

    double sum1 = ( a + b ) + c;
    double sum2 = a + ( b + c );

    std::cout << std::setprecision(20) << sum1 << std::endl <<
                 std::setprecision(20) << sum2 << std::endl;

    if (sum1 == .6)
      std::cout << "(.1 + .2) + .3 == .6" << std::endl;

    if (sum2 == .6)
      std::cout << ".1 + (.2 + .3) == .6" << std::endl;

    float minf = std::numeric_limits<float>::min();
    float smallf = 1.01f * minf;

    std::cout << "\n\n";
    std::cout << "Val1: " << std::setprecision(20) << smallf << std::endl;
    std::cout << "Val2: " << std::setprecision(20) << minf << std::endl;
    std::cout << "With denormals: " << std::endl;
    std::cout << "Val1 - Val2: " << smallf - minf << std::endl;

    enableFtzDas();
    
    std::cout << std::endl << "Without denormals: " << std::endl;
    std::cout << "Val1 - Val2: " << smallf - minf << std::endl;

    return 0;
}
