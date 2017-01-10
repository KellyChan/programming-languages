#include <iostream>

#include "gtest/gtest.h"

// #include "demo/foo/foo.h"


TEST (func_a, resetToZero)
{
    int i = 3;
//    func_a(i);
    EXPECT_EQ (0, i);

    i = 12;
//    func_a(i);
    EXPECT_EQ (0, i);
}


