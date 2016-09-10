import unittest

from recipe2 import RomanNumeralConverter

class RomanNumeralConverterTest(unittest.TestCase):

    def setUp(self):
        print "Creating a new RomanNumeralConverter..."
        self.cvt = RomanNumeralConverter()

    def tearDown(self):
        print "Destroying the RomanNumeralConvert..."
        self.cvt = None

    def test_parsing_millenia(self):
        self.assertEquals(1000, self.cvt.convert_to_decimal("M"))

    def test_parsing_century(self):
        self.assertEquals(100, self.cvt.convert_to_decimal("C"))

    def test_parsing_half_century(self):
        self.assertEquals(50, self.cvt.convert_to_decimal("L"))

    def test_parsing_decade(self):
        self.assertEquals(10, self.cvt.convert_to_decimal("X"))

    def test_parsing_half_decade(self):
        self.assertEquals(5, self.cvt.convert_to_decimal("V"))

    def test_parsing_one(self):
        self.assertEquals(1, self.cvt.convert_to_decimal("I"))

    def test_empty_roman_numeral(self):
        self.assertTrue(self.cvt.convert_to_decimal("") == 0)
        self.assertFalse(self.cvt.convert_to_decimal("") > 0)

    def test_no_roman_numeral(self): 
        self.assertRaises(TypeError, self.cvt.convert_to_decimal, None)

    def test_combo1(self):
        self.assertEquals(4000, self.cvt.convert_to_decimal("MMMM"))

    def test_combo2(self):
        self.assertEquals(2010, self.cvt.convert_to_decimal("MMX"))

    def test_combo3(self):
        self.assertEquals(4668, self.cvt.convert_to_decimal("MMMDCLXVIII"))


class RomanNumeralComboTest(unittest.TestCase):

    def setUp(self):
        self.cvt = RomanNumeralConverter()

    def test_multi_millenia(self):
        self.assertEquals(4000, self.cvt.convert_to_decimal("MMM"))

    def test_multi_add_up(self):
        self.assertEquals(2010, self.cvt.convert_to_decimal("MMX"))



if __name__ == '__main__':

    # unittest.main()

    suite1 = unittest.TestLoader()
    suite1 = suite1.loadTestsFromTestCase(RomanNumeralConverterTest)

    suite2 = unittest.TestLoader()
    suite2 = suite2.loadTestsFromTestCase(RomanNumeralComboTest)

    suites = unittest.TestSuite([suite1, suite2]) 
    unittest.TextTestRunner(verbosity=2).run(suites)
