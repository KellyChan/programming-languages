#include <include/lists.h>

bool find(List<int>& L, int K)
{
    int it;
    for (L.movetoStart(); L.rightLength() > 0; L.next())
    {
        it = L.getValue();
        if (K == it) return true;
    }
    return false;
}
