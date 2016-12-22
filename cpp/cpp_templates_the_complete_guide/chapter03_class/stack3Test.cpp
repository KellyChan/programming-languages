#include <deque>
#include <cstdlib>
#include <iostream>

#include "stack3.hpp"


int main()
{
    try
    {
        // stack of ints
        Stack<int> intStack;

        // stack of doubles which uses a std::deque<> to manage the elements
        Stack<double, std::deque<double> > dblStack;

        // manipulate int stack
        intStack.push(7);
        std::cout << intStack.top() << std::endl;
        intStack.pop();

        // manipulate double stack
        dblStack.push(42.42);
        std::cout << dblStack.top() << std::endl;
        dblStack.pop();
        dblStack.pop();
    }
    catch (std::exception const& ex)
    {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;  // exit program with ERROR status
    }
}
