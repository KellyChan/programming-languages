#include <string>
#include <sstream>
#include <iostream>


template <class T>
std::string dec2hex (T t)
{
    std::stringstream streamer;
    streamer << std::hex << t;
    return streamer.str();
}


template <class T>
T hex2dec (std::string h)
{
    T t;

    std::stringstream streamer;
    streamer << std::dec << h;
    streamer >> t;
    return t;
}


int main()
{
    std::cout << dec2hex<long>(2047) << std::endl;
    std::cout << hex2dec<int>("0x3") << std::endl;

    return 0;
}
