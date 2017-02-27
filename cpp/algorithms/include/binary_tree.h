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


// binary search tree implementation for the dictionary ADT
template <typename Key, typename Elem>
class BST : public Dictionary<Key, Elem>
{
    private:
        BNode<Key, Elem>* root;    // root of the BST
        int nodecount;             // number of nodes in the BST

        void clearhelp (BNode<Key, Elem>*);
        BNode<Key, Elem>* inserthelp(BNode<Key, Elem>*, const Key&, const Elem&);
        BNode<Key, Elem>* deletemin(BNode<Key, Elem>*, BNode<Key, Elem>*&);
        BNode<Key, Elem>* removehelp(BNode<Key, Elem>*, const Key&, BNode<Key, Elem>*&);
        bool findhelp(BNode<Key, Elem>*, const Key&, Elem&) const;
        void printhelp(BNode<Key, Elem>*, int) const;


    public:
        BST ()
        {
            root = NULL;
            nodecount = 0;
        }

        ~BST() { clearhelp(root); }

        void clear ()
        {
            clearhelp(root);
            root = NULL;
            nodecount = 0;
        }

        void insert (const Key& k, const Elem& it)
        {
            root = inserthelp(root, k, it);
            nodecount++;
        }


        bool remove(const Key& K, Elem& it)
        {
            BNode<Key, Elem>* t = NULL;
            root = removehelp(root, K, t);
            if (t == NULL) return false;
            it = t->val();
            nodecount--;
            delete t;
            return true;
        }

        Elem removeAny ()
        {
            Assert (root != NULL, "Empty tree");
            BNode<Key, Elem>* t;
            root = deletemin(root, t);
            Elem it = t->val();
            delete t;
            nodecount--;
            return it;
        }


        bool find(const Key&K, Elem& it) const
        {
            return findhelp(root, K, it);
        }

        int size() { return nodecount; }

        void print() const
        {
            if (root == NULL) cout << "The BST is empty.";
            else printhelp(root, 0);
        }
};


template <typename Key, typename Elem>
bool BST<Key, Elem>::findhelp (BNode<Key, Elem>* root, const Key& K, Elem& e) const
{
    if (root == NULL) return false;
    else if (K < root->key())
        return findhelp(root->left(), K, e);
    else if (K > root->key())
        return findhelp(root->right(), K, e);
    else { e = root->val(); return true; }
}


template <typename Key, typename Elem>
BNode<Key, Elem>* BST<Key, Elem>::inserthelp (BNode<Key, Elem>* root, const Key& k, const Elem& it)
{
    if (root == NULL)
        return new BNode<Key, Elem>(k, it, NULL, NULL);
    if (k < root->key())
        root->setLeft(inserthelp(root->left(), k, it));
    else root->setRight(inserthelp(root->right(), k, it));
    return root;
}


template <typename Key, typename Elem>
BNode<Key, Elem>* BST<Key, Elem>::deletemin(BNode<Key, Elem>* root, BNode<Key, Elem>*& S)
{
    if (root->left() == NULL)
    {
        S = root;
        return root->right();
    }
    else
    {
        root->setLeft(deletemin(root->left(), S);
        return root;
    }
}


template <typename Key, typename Elem>
BNode<Key, Elem>* BST<Key, Elem>::removehelp(BNode<Key, Elem>* root, const Key& K, BNode<Key, Elem>*& R)
{
    if (root == NULL) return NULL;
    else if (K < root->key())
        root->setLeft(removehelp(root->left(), K, R));
    else if (K > root->key())
        root->setRight(removehelp(root->right(), K, R));
    else
    {
        BNode<Key, Elem>* temp;
        R = root;
        if (root->left() == NULL)
            root = root->right();
        else if (root->right() == NULL)
            root = root->left();
        else
        {
            root->setRight(deletemin(root->right(), temp));
            Elem te = root->val();
            root->setVal(temp->val());
            temp->setVal(te);
            R = temp;
        }
    }
    return root;
}


template <typename Key, typename Elem>
void BST<Key, Elem>::clearhelp(BNode<Key, Elem>* root)
{
    if (root == NULL) return;
    clearhelp(root->left());
    clearhelp(root->right());
    delete root;
}


template <typename Key, typename Elem>
void BST<Key, Elem>::printhelp(BNode<Key, Elem>* root, int level) const
{
    if (root == NULL) return;
    printhelp(root->left(), level+1);
    for (int i = 0; i < level; i++)
        cout << " ";
    cout << root->key() << "\n";
    printhelp(root->right(), level+1);
}
