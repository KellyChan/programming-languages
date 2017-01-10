#include <iostream>

#include "foo/foo.h"


int main ()
{
    std::cout << "Foo:" << std::endl;

    int x = 4;
    std::cout << x << std::endl;
    func_a(x);

    Foo foo;
    foo.func_b(x);
    std::cout << x << std::endl;
}
