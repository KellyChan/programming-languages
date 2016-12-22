template <typename T>
template <typename T2>
Stack<T>& Stack<T>::operator= (Stack<T2> const& op2)
{
    if ((void*)this == (void*)&op2) { return *this; }  // assignment to itself?

    Stack<T2> tmp(op2);   // create a copy of the assigned stack

    elems.clear();  // remove existing elements
    while (!tmp.empty())
    {
        elems.push_front(tmp.top());
        tmp.pop();
    }
    return *this;
}


template <typename T, typename CONT>
template <typename T2, typename CONT2>
Stack<T, CONT>&
Stack<T, CONT>::operator= (Stack<T2,CONT2> const& op2)
{
    if ((void*)this == (void*)&op2)
    {
        return *this;  // assignment to itself?
    }

    Stack<T2> tmp(op2);  // create a copy of the assigned stack

    elems.clear();
    while (!tmp.empty())
    {
        elems.push_front(tmp.top());
        tmp.pop();
    }
    return *this;
}
