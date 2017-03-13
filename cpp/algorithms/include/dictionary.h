// the dictionary abstract class
template <typename Key, typename Elem>
class Dictionary
{
    private:
        void operator = (const Dictionary&) {}
        Dictionary (const Dictionary&) {}

    public:
        Dictionary () {}
        virtual ~Dictionary() {}

        virtual void clear () = 0;
        virtual void insert (const Key& k, const Elem& e) = 0;
        virtual bool remove (const Key& k, Elem& e) = 0;

        virtual Elem removeAny() = 0;
        virtual bool find(const Key& k, Elem& e) const = 0;
        virtual int size () = 0;
};
