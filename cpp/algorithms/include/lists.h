// List ADT
template <typename Elem> class List
{
    private:
        // protect assignment
        void operator = (const List&) {}
        // protect copy constructor
        List(const List&) {}

    public:
        // default constructor
        List () {}
        // base destructor
        virtual ~List() {}

        // reinitialize the list
        // the client is responsible for reclaiming the storage
        // used by the list elements
        virtual void clear () = 0;

        // insert an element at the front ofo the right partition
        // return true if the insert was successful, false otherwise
        virtual bool insert (const Elem&) = 0;

        // append an element at the end of the right partition
        // return true if the append was successful, false otherwise
        virtual bool append (const Elem&) = 0;

        // remove and return the first element of right partition
        virtual Elem remove () = 0;

        // place fence at list start, making left partition empty
        virtual void movetoStart () = 0;

        // place fence at list end, making right partition empty
        virtual void movetoEnd () = 0;

        //move fence one step left, no change if at beginning
        virtual void prev () = 0;

        // move fence one step right, no change if already at end
        virtual void next () = 0;

        // return length of left or right partition, respectively
        virtual int leftLength () const = 0;
        virtual int rightLength () const = 0;

        // set fence so that left partition has "pos" elements
        virtual void movetoPos (int pos) = 0;

        // return the first element of the right partition
        virtual const Elem& getValue () const = 0;
};


// Array-based list implementation
template <typename Elem>
class AList : public List<Elem>
{
    private:
        int maxSize;       // maximum size of list
        int listSize;      // actual number of elements in list
        int fence;         // position of fence
        Elem* listArray;   // Array holding list elements

    public:
        // Constructor
        AList (int size=DefaultListSize)
        {
            maxSize = size;
            listSize = fence = 0;
            listArray = new Elem[maxSize];
        }

        // Destructor
        ~AList ()
        {
            delete [] listArray;
        }

       // reinitialize the list
       void clear ()
       {
           // remove the array
           delete [] listArray;
           // reset the size
           listSize = fence = 0;
           // recreate array
           listArray = new Elem[maxSize];
       }

       // insert "it" at front of right partition
       bool insert (const Elem& it)
       {
           if (listSize < maxSize)
           {
               for (int i=listSize; i > fence; i--)
                   listArray[i] = listArray[i-1];
               listArray[fence] = it;
               listSize++;
               return true;
           }
           else
               return false;
       }

       
       bool append (const Elem& it)
       {
           if (listSize < maxSize)
           {
               listArray[listSize++] = it;
               return true;
           }
           else
               return false;
       }

       // remove and return first Elem in right partition
       Elem remove ()
       {
           Assert (rightLength() > 0, "Nothing to remove");
           Elem it = listArray[fence];
           for (int i = fence; i < listSize-1; i++)
               listArray[i] = listArray[i+1];
           listSize--;
           return it;
       }

       void movetoStart () { fence = 0; }
       void movetoEnd () { fence = listSize; }
       void prev () { if (fence != 0) fence--; }
       void next() { if (fence < listSize) fence++; }

      // return left or right partition size
      int leftLength () const { return fence; }
      int rightLength () const { return listSize - fence; }

      void movetoPos (int pos)
      {
          Assert ((pos>=0) && (pos<=listSize), "Pos out of range");
          fence = pos;
      }

      const Elem& getValue () const 
      {
          Assert (rightLength () > 0, "Nothing to get");
          return listArray[fence];
      }
};


// singly linked list node
template <typename Elem> class Link
{
    public:
        Elem element;
        Link *next;

        // constructors
        Link (const Elem& elemval, Link* nextval=NULL)
        {
            element = elemval;
            next = nextval;
        }

        Link (Link* nextval=NULL)
       {
           next = nextval; 
       }
};


// Linked list implementation
template <typename Elem> class LinkedList : public List <Elem>
{
    private:
        Link<Elem>* head;       // pointer to list header
        Link<Elem>* tail;       // pointer to last elem in list
        Link<Elem>* fence;      // last element on left side
        int leftcnt;            // size of left partition
        int rightcnt;           // size of right partition

        void init ()
        {
            fence = tail = head = new Link<Elem>;
            leftcnt = rightcnt = 0;
        }

        void removeall ()
        {
            while (head != NULL)
            {
                fence = head;
                head = head->next;
                delete fence;
            }
        }


    public:
        // constructor
        LinkedList (int size=DefaultListSize) { init (); }
        // destructor
        ~LinkedList () { removeall(); }
        // print list contents
        void print () const;
        // clear list
        void clear () { removeall(); init(); }

        // insert "it" at front of right partition
        bool insert (const Elem& it)
        {
            fence->next = new Link<Elem> (it, fence->next);
            if (tail == fence) tail = fence->next;  // new tail
            rightcnt++;
            return true;
        }

        bool append (const Elem& it)
        {
            tail = tail->next = new Link<Elem> (it, NULL);
            rightcnt++;
            return true;
        }

        // place fence at list start
        bool movetoStart ()
        {
            fence = head;
            rightcnt += leftcnt;
            leftcnt = 0;
        }

        void movetoEnd ()
        {
            fence = tail;
            leftcnt += rightcnt;
            rightcnt = 0;
        }

       // remove and return first Elem in right partition
       Elem remove ()
       {
           Assert (rightLength () > 0, "Nothing to remove");
           Elem it = fence->next->element;    // remember value
           Link<Elem>* ltemp = fence->next;   // remember link node
           fence->next = ltemp->next;         // remove from list
           if (tail == ltemp) tail = fence;   // reset tail
           delete ltemp;                      // reclaim space
           rightcnt--;                        // decrement the count

           return it;
       }

       // move fence one step left; no change if left is empty
       void prev ()
       {
           if (fence == head) return;         // no previous Elem
           Link<Elem>* temp = head;
           // March down list until we find the previous element
           while (temp->next != fence) temp = temp->next;
           fence = temp;
           leftcnt--;
           rightcnt++;
       }

       // move fence one step right; no change if right is empty
       void next ()
       {
           if (fence != tail)
           {
               fence = fence->next;
               rightcnt--;
               leftcnt++;
           }
       }

       int leftLength () const { return leftcnt; }
       int rightLength () const { return rightcnt; }

       // set the size of left partition to "pos"
       void movetoPos (int pos)
       {
           Assert ((pos >= 0) && (pos <= rightcnt + leftcnt), "Position out of range");
          rightcnt = rightcnt + leftcnt - pos;
          leftcnt = pos;
          fence = head;
          for (int i = 0; i < pos; i++) fence = fence->next;
       }

 
       const Elem& getValue () const 
       {
           Assert (rightLength () > 0, "Nothing to get");
           return fence->next->element;
       }
};


// Singly linked list node with freelist support
template <typename Elem> class Link
{
    private:
        // reference to freelist head  
        static Link<Elem>* freelist;

    public:
        Elem element;
        Link* next;

        // constructors
        Link (const Elem& elemval, Link* nextval=NULL)
        {
            element = elemval;
            next = nextval;
        }

        Link (Link* nextval=NULL)
        {
            next = nextval;
        }

        // overloaded new operator
        void* operator new (size_t)
        {
            // create space
            if (freelist == NULL) return ::new Link;
            // can take from freelist
            Link<Elem>* temp = freelist;
            freelist = freelist->next;
            return temp;
        }

        // overloaded delete operator
        void operator delete (void* ptr)
        {
            ((Link<Elem>*)ptr)->next = freelist;  // put on freelist
            freelist = (Link<Elem>*)ptr;
        }
};


// The freelist head pointer is actually created here
template <typename Elem>
Link<Elem>* Link<Elem>::freelist = NULL;


// doubly linked list node with freelist support
template <typename Elem> class Link
{
    private:
        // reference to freelist head
        static Link<Elem>* freelist;

    public:
        Elem element;    // value for this node
        Link* next;      // pointer to next node in list
        Link* prev;      // pointer to previous node

        // constructors
        Link (const Elem& e, Link* prevp, Link* nextp)
        {
            element = e;
            prev = prevp;
            if (prev != NULL) prev->next = this;
            next = nextp;
            if (next != NULL) next->prev = this;
        }

        Link (Link* prevp=NULL, Link* nextp=NULL)
        {
            prev = prevp;
            if (prev != NULL) prev->next = this;
            next = nextp;
            if (next != NULL) next->prev = this;
        }

        // overloaded new operator
        void* operator new(size_t)
        {
            // create space
            if (freelist == NULL) return ::new Link;
            Link<Elem>* temp = freelist;
            freelist = freelist->next;
            return temp;
        }

        // overloaded delete operator
        void operator delete (void* ptr)
        {
            // put ono freelist
            ((Link<Elem>*ptr)->next = freelist;
            freeList = (Link<Elem>*)ptr;
        }
};

// The freelist head pointer is actually created here
template <typename Elem>
Link<Elem>* Link<Elem>::freelist = NULL;
