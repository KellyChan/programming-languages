#include "unity.h"

#include "demo.h"


void setUp(void)
{
  //
}


void tearDown(void)
{
  //
}


void test_demo(void)
{
  int a = 1;
  int b = 3;
  int c;
  int expected = 4;
  c = add(a, b);
  TEST_ASSERT_EQUAL_INT(expected, c);
}


int main()
{
  UNITY_BEGIN();
  RUN_TEST(test_demo);
  return UNITY_END();
}
