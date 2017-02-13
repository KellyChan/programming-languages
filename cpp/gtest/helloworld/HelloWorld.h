#ifndef HELLOWORLD_H_
#define HELLOWORLD_H_

#include <string>

#include "Messenger.h"

using namespace std;


class HelloWorld
{
    public:
        HelloWorld();
        virtual ~HelloWorld();
        string getMessage(Messenger* messenger) const;
};

#endif  // HELLOWORLD_H
