// binary tree node abstract class
template <typename Elem> class BinNode
{
    public:
        virtual ~BinNode () {}

        virtual Elem& val() = 0;
        virtual void setVal(const Elem&) = 0;

        virtual BinNode* left() const = 0;
        virtual void setLeft(BinNode*) = 0;
        virtual BinNode* right() const = 0;
        virtual void setRight(BinNode*) = 0;

        virtual bool isLeaf() = 0;
};


template <typename Elem>
void preorder(BinNode<Elem>* root)
{
    if (root == NULL) return;   // empty subtree, do nothing
    visit(root);
    preorder(root->left());
    preorder(root->right());
}


template <typename Elem>
void preorder2(BinNode<Elem>* root)
{
    visit(root);
    if (root->left() != NULL) preorder2(root->left());
    if (root->right() != NULL) preorder2(root->right());
}


template <typename Elem>
int count (BinNode<Elem>* root)
{
    if (root == NULL) return 0;
    return 1 + count(root->left()) + count(root->right());
}


template <typename Key, typename Elem>
bool checkBST (BNode<Key, Elem>* root, Key, low, Key high)
{
    if (root == NULL) return true;
    Key rootkey = root->key();
    if ((rootkey < low) || (rootkey > high))
        return false;

    if (!checkBST<Key, Elem>(root->left(), low, rootkey))
        return false;

    return checkBST<Key, Elem>(root->right(), rootkey, high);
}


// simple binary tree node implementation
template <typename Key, typename Elem>
class BNode: public BinNode<Elem>
{
    private:
        Key k;
        Elem it;
        BNode* lc;
        BNode* rc;

    public:
        BNode () { lc = rc = NULL; }
        BNode (Key k, Elem e, BNode* l=NULL, BNode* r=NULL)
        {
            k = K;
            it = e;
            lc = l;
            rc = r;
        }

        ~BNode () {}

        // Functions to set and return the value and key
        Elem& val() { return it; }
        void setVal (const Elem& e) { it = e; }
        Key& key() { return k; }
        void setKey (const Key& K) { k = K; }

        // functions to set and return the children
        inline BNode* left() const { return lc; }
        void setLeft (BinNode<Elem>* b) { lc = (BNode*)b; }
        inline BNode* right() const { return rc; }
        void setRight (BinNode<Elem>* b) { rc = (BNode*) b; }

        // return true if it is a leaf, false otherwise
        bool isLeaf() { return (lc == NULL) && (rc == NULL); }
};


// node implementation with simple inheritance
class VarBinNode
{
    public:
        virtual ~VarBinNode() {}
        virtual bool isLeaf() = 0;
};


class LeafNode: public VarBinNode
{
    private:
        Operand var;

    public:
        LeafNode(const Operand& val) { val = val; }
        bool isLeaf() { return true; }
        Operand value() { return var; }
};


class IntlNode: public VarBinNode
{
    private:
        VarBinNode* left;
        VarBinNode* right;
        Operator opx;

    public:
        IntlNode (const Operator& op, VarBinNode* l, VarBinNode* r)
        {
            opx = op;
            left = l;
            right = r;
        }
   
        bool isLeaf() { return false; }
        VarBinNode* leftchild() { return left; }
        VarBinNode* rightchild() { return right; }
        Opeartor value() { return opx; }
};


void traverse (VarBinNode * root)
{
    if (root == NULL) return;
    if (root->isLeaf())
        cout << "Leaf: " << ((LeafNode *)root)->value() << endl;
    else
    {
        cout << "Leaf: " << ((IntlNode*)root)->value() << endl;
        traverse(((IntlNode*)root)->leftchild());
        traverse(((IntlNode*)root)->rightchild());
    }
};


// node implementation with the compositie design pattern
class VarBinNode
{
    public:
        virtual ~VarBinNode () {}
        virtual bool isLeaf() = 0;
        virtual void traverse() = 0;
};


class LeafNode: public VarBInNode
{
    private:
        Operand var;

     public:
        LeafNode(const Operand& val) { var = val; }
        bool isLeaf() { return true; }
        Operand value() { return var; }
        void traverse() { cout << "Leaf: " << value() << endl;
};


class IntlNode:public VarBinNode
{
    private:
        VarBinNode* lc;
        VarBinNode* rc;
        Operator opx;

    public:
        IntlNode (const Operator& op, VarBinNode* l, VarBinNode* r)
        {
            opx = op;
            lc = l;
            rc = r;
        }

        bool isLeaf() { return false; }
        VarBinNode* left() { return lc; }
        VarBinNode* right() { return rc; }
        Operator value() { return opx; }

        void traverse() 
        {
            cout << "Internal: " << value() << endl;
            if (left() != NULL) left()->traverse();
            if (right() != NULL) right()->traverse();
        }
};


// do a preorder traversal
void traverse(VarBinNode *root)
{
    if (root != NULL) root->traverse();
}
