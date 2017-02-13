#ifndef MOCKMESSENGER_H_
#define MOCKMESSENGER_H_

#include <string>
#include <gmock/gmock.h>

#include "Messenger.h"

class MockMessenger:public Messenger
{
    public:
        MOCK_METHOD0(getMessage, string());
};

#endif  // MOCKMESSENGER_H_
