// stack abtract class
template <typename Elem> class Stack
{
    private:
        // protect assignment
        void operator = (const Stack&) {}
        // protect copy constructor
        Stack (const Statck&) {}

   public:
        // default constructor
        Stack () {}
        // base destructor
        virtual ~Stack() {}

        // reinitialize the stack.
        // the use is responsible for reclaiming
        // the storage used by the stack elements
        virtual void clear () = 0;

        // push an element onto the top of the stack
        virtual void push (const Elem&) = 0;

        // remove and return the element at the top of the stack
        virtual Elem pop () = 0;

        // return a copy of the top element
        virtual const Elem& topValue() const = 0;

        // return the number of elements in the stack
        virtual int length() const = 0;
};


// array-based stack implementation
template <typename Elem> class Stack : public Stack<Elem>
{
    private:
        int maxSize;          // maximum size of stack
        int top;              // index for top element
        Elem *listArray;      // Array holding stack elements

    public:
        // constructor
        AStack (int size = DefaultListSize)
        {
            maxSize = size;
            top = 0;
            listArray = new Elem[size];
        }

        // destructor
        ~AStack()
       {
            delete [] listArray;
       }

        // reinitialize
        void clear () { top = 0; }

        void push (const Elem& it)
        {
            Assert(top != maxSize, "Stack is full");
            listArray[top++] = it;
        }

        Elem pop()
        {
            Assert (pop != 0, "Stack is empty");
            return listArray[--top];
        }

        const Elem& topValue() const
        {
            Assert (top != 0, "Stack is empty");
            return listArray[top-1];
        }

        int length() const { return top; }
};


// linked stack implementation
template <typename Elem> class LStack: public Stack<Elem>
{
    private:
        Link<Elem>* top;     // pointer to first element
        int size;            // number of elements

    public:
        LStack (int sz = DefaultListSize)
        {
            top = NULL;
            size = 0;
        }

        ~LStack() { clear(); }

        void clear()
        {
            while (top != NULL)
            {
                Link<Elem>* temp = top;
                top = top->next;
                size = 0;
                delete temp;
            }
        }

        void push (const Elem& it)
        {
            top = new Link<Elem>(it, top);
            size++;
        }

        Elem pop()
        {
            Assert(top != NULL, "Stack is empty");
            Elem it = top->element;
            Link<Elem>* ltemp = top->next;
            delete top;
            top = ltemp;
            size--;
            return it;
        }

        const Elem& topValue() const
        {
            Assert(top != 0, "Stack is empty");
            return top->element;
        }

        int length() const { return size; }
};
