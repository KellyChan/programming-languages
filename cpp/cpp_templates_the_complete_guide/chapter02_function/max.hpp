#ifndef MAX_H
#define MAX_H

template <typename T>
inline T const& max (T const& a, T const& b)
{
    // if a < b then use b else a
    return a < b ? b : a;
}

#endif
