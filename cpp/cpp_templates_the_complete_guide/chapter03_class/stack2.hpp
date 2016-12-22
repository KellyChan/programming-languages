#include <deque>
#include <string>
#include <stdexcept>

#include "stack.hpp"


template <>
class Stack<std::string>
{
    private:
        std::deque<std::string> elems;  // elements

    public:
        void push(std::string const&);  // push element
        void pop();                     // pop element
        std::string top() const;        // return top element
        bool empty() const { return elems.empty(); }  // return whether the stack is empty
};


void Stack<std::string>::push (std::string const& elem)
{
    elems.push_back(elem);  // append copy of passed elem
}


void Stack<std::string>::pop ()
{
    if (elems.empty())
    {
        throw std::out_of_range("Stack<std::string>::pop(): empty stack");
    }
    elems.pop_back();  // remove last element
}


std::string Stack<std::string>::top () const
{
    if (elems.empty())
    {
        throw std::out_of_range("Stack<std::string>::top(): empty stack");
    }
    return elems.back();  // return copy of last element
}
