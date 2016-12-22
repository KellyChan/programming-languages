#include <vector>
#include <stdexcept>


template <typename T, typename CONT = std::vector<T> >
class Stack
{
    private:
        CONT elems;           // elements

    public:
        void push(T const&);          // push element
        void pop();                   // pop element
        T top() const;                // return top element
        bool empty() const { return elems.empty(); }  // return whether the stack is empty
};


template <typename T, typename CONT>
void Stack<T, CONT>::push (T const& elem)
{
    elems.push_back(elem);               // append copy of passed elem
}


template <typename T, typename CONT>
void Stack<T, CONT>::pop ()
{
    if (elems.empty())
    {
        throw std::out_of_range("Stack<>::pop(): empty stack");
    }
    elems.pop_back();  // remove last element
}


template <typename T, typename CONT>
T Stack<T, CONT>::top () const
{
    if (elems.empty())
    {
        throw std::out_of_range("Stack<>::top(): empty stack");
    }
    return elems.back();  // return copy of last element
}
