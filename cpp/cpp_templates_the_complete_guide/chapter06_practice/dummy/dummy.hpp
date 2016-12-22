#ifndef DUMMY_HPP
#define DUMMY_HPP

#include <iostream>
#include <typeinfo>


// declaration of template
template <typename T>
void print_typeof (T const&);


template <typename T>
void print_typeof (T const& x)
{
    std::cout << typeid(x).name() << std::endl;
}


#endif // DUMMY_HPP
