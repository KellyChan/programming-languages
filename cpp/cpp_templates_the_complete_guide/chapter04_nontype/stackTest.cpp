#include <string>
#include <cstdlib>
#include <iostream>

#include "stack.hpp"


int main()
{
    try
    {
        Stack<int,20> int20Stack;             // stack of up to 20 ints
        Stack<int,40> int40Stack;             // stack of up to 40 ints
        Stack<std::string,40> stringStack;    // stack of up to 40 strings

        // manipulate stack of up to 20 ints
        int20Stack.push(7);
        std::cout << int20Stack.top() << std::endl;
        int20Stack.pop();

        // manipulate stack of up to 40 strings
        stringStack.push("hello");
        std::cout << stringStack.top() << std::endl;
        stringStack.pop();
        stringStack.pop();
    }
    catch (std::exception const& ex)
    {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;  // exit program with ERROR status
    }
}
