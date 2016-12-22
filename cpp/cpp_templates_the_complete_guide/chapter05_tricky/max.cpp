#include <string>

template <typename T>
inline T const& max (T const& a, T const& b)
{
    return a < b ? b : a;
}


inline T max (T a, T b)
{
    return a < b ? b : a;
}


int main()
{
    std::string s;

    ::max("apple", "peach");
    ::max("apple", s);
}
