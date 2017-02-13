#ifndef MESSENGER_H_
#define MESSENGER_H_

#include <string>

using namespace std;


class Messenger
{
    public:
        virtual ~Messenger(){};
        virtual string getMessage() = 0;
};

#endif  // MESSENGER_H_
