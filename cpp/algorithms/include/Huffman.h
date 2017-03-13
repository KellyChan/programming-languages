template <typename Elem>
class HuffNode
{
    public:
        virtual ~HuffNode () {}
        virtual int weight () = 0;
        virtual bool isLeaf () = 0;
        virtual HuffNode* left () const = 0;
        virtual void setLeft (HuffNode*) = 0;
        virtual HuffNode* right () const = 0;
        virtual void setRight (HuffNode*) = 0; 
};


template <typename Elem>
class LeafNode : public HuffNode<Elem>
{
    private:
        KVpair<int, Elem>* it;

    public:
        LeafNode (int freq, const Elem& val)
        {
            it = new KVpair<int, Elem>(freq, val);
        }
        int weight () { return it->key(); }
        KVpair<int, Elem>* val() { return it; }
        bool isLeaf() { return true; }
        virtual HuffNode<Elem>* left () const { return NULL; }
        virtual void setLeft(HuffNode<Elem>*) {}
        virtual HuffNode<Elem>* right () const { return NULL; }
        virtual void setRight (HuffNode<Elem>*) {}
};


template <typename Elem>
class IntlNode : public HuffNode<Elem>
{
    private:
        HuffNode<Elem>* lc;        // left child
        HuffNode<Elem>* rc;        // right child
        int wgt;                   // subtree weight

    public:
        IntlNode(HuffNode<Elem>* l, HuffNode<Elem>* r)
        {
            wgt = l->weight() + r->weight();
            lc = l;
            rc = r;
        }

        int weight() { return wgt; }
        bool isLeaf () { return false; }
        HuffNode<Elem>* left() const { return lc; }
        void setLeft(HuffNode<Elem>* b)
        {
            lc = (HuffNode<Elem>*) b;
        }

        HuffNode<Elem>* right () const { return rc; }
        void setRight (HuffNode<Elem>* b)
        {
            rc = (HuffNode<Elem>*) b;
        }
};


template <typename Elem>
class HuffTree
{
    private:
        HuffNode<Elem>* Root;

    public:
        HuffTree(Elem& val, int freq)
        {
            Root = new LeafNode<Elem>(freq, val);
        }

        HuffTree(HuffTree<Elem>* l, HuffTree<Elem>* r)
        {
            Root = new IntlNode<Elem>(l->root(), r->root());
        }

        ~HuffTree() {}

        HuffNode<Elem>* root() { return Root; }
        int weight() { return Root->weight(); }
};


template <typename Elem> HuffTree<Elem>* buildHuff(heap<HuffTree<Elem>*, minTreeComp>* buildHuff(heap<HuffTree<Elem>*, minTreeComp>* fl)
{
    HuffTree<Elem> *temp1, *temp2, *temp3 = NULL;
    while (fl->size() > 1)
    {
        temp1 = fl->removefirst();
        temp2 = fl->removefirst();
        temp3 = new HuffTree<Elem> (temp1, temp2);
        fl->insert(temp3);
        delete temp1;
        delete temp2;
    }
    return temp3;
}
