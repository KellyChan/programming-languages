// general tree node ADT
template <typename Elem>
class GTNode
{
    public:
        Elem value();
        bool isLeaf();
        GTNode* parent();
        GTNode* leftmostChild();
        GTNode* rightSibling();
        void setValue(Elem&);
        void insertFirst(GTNode<Elem>*);
        void insertNext(GTNode<Elem>*);
        void removeFirst();
        void removeNext();
};


// general tree ADT
template <typename Elem>
class GenTree
{
    public:
        void clear();
        GTNode<Elem>* root();
        void newroot(Elem&, GTNode<Elem>*, GTNode<Elem>*);
        void print();
};


// print using a preorder traversal
void printhelp(GTNode<Elem>* root)
{
    if (root->isLeaf()) cout << "Leaf: ";
    else cout << "Internal: ";
    cout << root->value() << "\n";
    for (GTNode<Elem>* temp = root->leftmostChild();
         temp != NULL;
         temp = temp->rightSibling())
        printhelp(temp);
}


// general tree representation for UNION/FIND
class ParPtrTree
{
    private:
        int* array;
        int size;
        int FIND(int) const;

    public:
        ParPtrTree(int);
        ~ParPtrTree() { delete [] array; }
        void UNION (int, int);
        bool differ(int, int);
};


PartPtrTree::ParPtrTree(int sz)
{
    size = sz;
    array = new int[sz];
    for (int i = 0; i < sz; i++) array[i] = ROOT;
}


// return True if nodes are in different trees
bool ParPtrTree::differ(int a, int b)
{
    int root1 = FIND(a);
    int root2 = FIND(b);
    return root1 != root2;
}


void ParPtrTree::UNIOIN(int a, int b)
{
    int root1 = FIND(a);
    int root2 = FIND(b);
    if (root1 != root2) array[root2] = root1;
}


int ParPtrTree::FIND(int curr) const
{
    while (array[curr] != ROOT) curr = array[curr];
    return curr;
}
