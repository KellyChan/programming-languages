template <typename T>
class Stack
{
    private:
        std::deque<T> elems;        // elements

    public:
        void push(T const&);        // push element
        void pop();                 // pop element
        T top() const;              // return top element
        bool empty() const { return elems.empty(); }  // return whether the stack is empty

    // assign stack of elements of type T2
    template <typename T2>
    Stack<T>& operator= (Stack<T2> const&);
};


template <typename T, typename CONT = std::deque<T> >
class Stack
{
    private:
        CONT elems;             // elements

    public:
        void push (T const&);    // push element
        void pop();              // pop element
        T top() const;           // return top element
        bool empty() const { return elems.empty(); }  // return whether the stack is empty

        // assign stack of elements of type T2
        template <typename T2, typename CONT2>
        Stack<T, CONT>& operator= (Stack<T2,CONT2> const&);
};


template <typename T, template <typename ELEM> class CONT = std::deque>
class Stack
{
    private:
        CONT<T> elems;   // elements

    public:
        void push(T const&);   // push element
        void pop();            // pop element
        T top() const;         // return top element
        bool empty() const { return elems.empty(); }  // return whether the stack is empty
}


// template <typename T, template <typename ELEM,
//                                 typename ALLOC = std::allocator<ELEM>> class CONT = std::deque>
