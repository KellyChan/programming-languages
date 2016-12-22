#include <stdexcept>


template <typename T, int MAXSIZE>
class Stack
{
    private:
        T elems[MAXSIZE];             // elements
        int numElems;                 // current number of elements

    public:
        Stack();                      // constructor
        void push(T const&);          // push element
        void pop();                   // pop element
        T top() const;                // return top element
        bool empty() const { return numElems == 0; }  // return whether the stack is empty
        bool full() const { return numElems == MAXSIZE; }  // return whether the stack is full
};


// constructor
template <typename T, int MAXSIZE>
Stack<T, MAXSIZE>::Stack(): numElems(0)  // start with no elements
{
    // nothing else to do
}


template <typename T, int MAXSIZE>
void Stack<T, MAXSIZE>::push (T const& elem)
{
    if (numElems == MAXSIZE)
    {
        throw std::out_of_range("Stack<>:push(): stack is full");
    }
    elems[numElems] = elem;  // append element
    ++numElems;              // increment number of elements
}


template <typename T, int MAXSIZE>
void Stack<T, MAXSIZE>::pop ()
{
    if (numElems <= 0)
    {
        throw std::out_of_range("Stack<>::pop(): empty stack");
    }
    --numElems;  // decrement number of elements
}


template <typename T, int MAXSIZE>
T Stack<T, MAXSIZE>::top () const
{
    if (numElems <= 0)
    {
        throw std::out_of_range("Stack<>::top(): empty stack");
    }
    return elems[numElems-1];  // return last element
}
