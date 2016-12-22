template <typename T, int VAL>
T addValue (T const& x)
{
    return x + VAL;
}


std::transform (source.begin(), source.end(), // start and end of source
                dest.begin(),                 // start of destination
                addValue<int,5>);             // operation


std::transform (source.begin(), source.end(),  // start and end of source
                dest.begin(),                  // start of destination
                (int(*)(int const&) addValue<int,5>);  // operation
