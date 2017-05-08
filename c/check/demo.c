#include <check.h>

START_TEST (test_demo)
{
  ck_assert_int_eq(1,1);
  ck_assert_str_eq("USD", "USD");
}
END_TEST

int main(void)
{
  return 0;
}
