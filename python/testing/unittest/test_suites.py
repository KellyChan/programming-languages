import unittest

from recipe2 import RomanNumeralConverter
from test_recipe2 import RomanNumeralConverterTest

def high_and_low():
    # an example of testing the edges
    suite = unittest.TestSuite()
    suite.addTest(RomanNumeralConverterTest("test_parsing_millenia"))
    suite.addTest(RomanNumeralConverterTest("test_parsing_one"))
    return suite
 
def combos():
    # a random sampling of values used to show that things are generally working
    tests = ['test_combo1', 'test_combo2', 'test_combo3']
    suites = map(RomanNumeralConverterTest, tests)
    return unittest.TestSuite(suites)

def all():
    # the comprehensive suite
    return unittest.TestLoader().loadTestsFromTestCase(RomanNumeralConverterTest)

if __name__ == '__main__':

    for suite_func in [high_and_low, combos, all]:
        print "Running test suite '%s'" % suite_func.func_name
        suite = suite_func()
        # print suite
        unittest.TextTestRunner(verbosity=2).run(suite)
