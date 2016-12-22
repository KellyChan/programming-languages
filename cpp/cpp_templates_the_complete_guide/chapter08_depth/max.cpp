template <typename T>
inline T const& max (T const& a, T const& b)
{
    return a < b ? b : a;
}


int main()
{
    max<double>(1.0, -3.0);
    max(1.0, -3.0);
    max<int>(1.0, 3.0);
}
