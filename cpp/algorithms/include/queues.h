// abstract queue class
template <typename Elem> class Queue
{
    private:
        void operator = (const Queue&) {}
        Queue (const Queue&) {}

    public:
        Queue () {}
        virtual ~Queue () {}

        virtual void clear () = 0;
        virtual void enqueue (const Elem&) = 0;
        virtual Elem dequeue () = 0;
        virtual const Elem& frontValue() const = 0;
        virtual int length() const = 0;
};


// array-based queue implementation
template <typename Elem> class AQueue: public Queue<Elem>
{
    private:
        int maxSize;
        int front;
        int rear;
        Elem *listArray;

    public:
        AQueue(int size=DefaultListSize)
        {
            maxSize = size+1;
            rear = 0;
            front = 1;
            listArray = new Elem[maxSize];
        }

        ~AQueue() { delete [] listArray; }

        void clear ()
        {
            rear = 0;
            front = 1;
        }

        void enqueue (const Elem& it)
        {
            Assert(((rear+2) % maxSize) != front, "Queue is full");
            rear = (rear + 1) % maxSize;
            listArray[rear] = it;
        }

        Elem dequeue ()
        {
            Assert(length () != 0, "Queue is empty");
            Elem it = listArray[front];
            front = (front+1) % maxSize;
            return it;
        }

        const Elem& frontValue() const
        {
            Assert(length() != 0, "Queue is empty");
            return listArray[front];
        }

        virtual int length() const
        {
            return ((rear+maxSize) - front + 1) % maxSize;
        }
};


// linked queue implementation
template <typename Elem> class LQueue: public Queue<Elem>
{
    private:
        Link<Elem>* front;
        Link<Elem>* rear;
        int size;

    public:
        LQueue (int sz = DefaultListSize)
        {
            front = rear = new Link<Elem>();
            size = 0;
        }

        ~LQueue () 
        {
            clear ();
            delete front;
       }

       void clear ()
       {
           while (front->next != NULL)
           {
               rear = front;
               delete rear;
           }
           rear = front;
           size = 0;
       }

       void enqueue (const Elem& it)
       {
           rear->next = new Link<Elem> (it, NULL);
           rear = rear->next;
           size++; 
       }

       Elem dequeue()
       {
            Assert(size != 0, "Queue is empty");
            Elem it = front->next->element;
            Link<Elem>* ltemp = front->next;
            front->next = ltemp->next;
            if (rear == ltemp) rear = front;
            delete ltemp;
            size--;
            return it;
       }

       const Elem& frontValue () const
       {
           Assert(size != 0, "Queue is empty");
           return front->next->element;
       }

       virtual int length() const { return size; }
};
