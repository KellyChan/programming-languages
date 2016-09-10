import unittest

from recipe2 import RomanNumeralConverter
from test_recipe2_legacy import RomanNumeralTester


if __name__ == '__main__':

    tester = RomanNumeralTester()
    suite = unittest.TestSuite()

    tests = [tester.simple_test, \
             tester.combo_test1, \
             tester.combo_test2, \
             tester.other_test]
    for test in tests:
        testcase = unittest.FunctionTestCase(test)
        suite.addTest(testcase)

    unittest.TextTestRunner(verbosity=2).run(suite)

