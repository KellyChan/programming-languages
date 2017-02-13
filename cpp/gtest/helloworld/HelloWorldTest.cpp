#include <string>
#include <memory>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "HelloWorld.h"
#include "MockMessenger.h"


using namespace testing


TEST (HelloWorldTest, getMessage)
{
    MockMessenger messenger;
    std::string msg = "Hello World";
    EXPECT_CALL(messenger, getMessage()).WillRepeatedly(Return(ByRef(msg)));

    HelloWorld helloWorld;
    EXPECT_EQ("Hello World", helloWorld.getMessage(&messenger));
    EXPECT_EQ("Hello World", helloWorld.getMessage(&messenger);
    EXPECT_EQ("Hello World", helloWorld.getMessage(&messenger);
}
