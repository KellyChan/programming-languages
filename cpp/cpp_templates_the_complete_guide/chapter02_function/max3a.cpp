#include <string>
#include <cstring>
#include <iostream>


template <typename T>
inline T const& max (T const& a, T const& b)
{
    return a < b ? b : a;
}


inline char const* max (char const* a, char const* b)
{
    return std::strcmp(a, b) < 0 ? b : a;
}


template <typename T>
inline T const& max (T const& a, T const& b, T const& c)
{
    return max(max(a,b),c);  // error, if max(a,b) use call-by-value
}


int main()
{
    ::max(7, 42, 49); // OK

    const char* s1 = "frederic";
    const char* s2 = "anica";
    const char* s3 = "lucas";
    ::max(s1, s2, s3);  // error
}
