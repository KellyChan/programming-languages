import unittest

from recipe3 import RomanNumeralConverter

class RomanNumeralTest(unittest.TestCase):

    def setUp(self):
        self.cvt = RomanNumeralConverter()

    def test_to_roman_bottom(self):
        self.assertEquals("I", self.cvt.convert_to_roman(1))

    def test_to_roman_below_botton(self):
        self.assertEquals("", self.cvt.convert_to_roman(0))

    def test_to_roman_negative_value(self):
        self.assertEquals("", self.cvt.convert_to_roman(-1))

    def test_to_roman_top(self):
        self.assertEquals("MMMM", self.cvt.convert_to_roman(4000))

    def test_to_roman_above_top(self):
        self.assertRaises(Exception, self.cvt.convert_to_roman, 4001)

    def test_to_decimal_bottom(self):
        self.assertEquals(1, self.cvt.convert_to_decimal("I"))

    def test_to_decimal_below_bottom(self):
        self.assertEquals(0, self.cvt.convert_to_decimal(""))

    def test_to_decimal_top(self):
        self.assertEquals(4000, self.cvt.convert_to_decimal("MMMM"))

    def test_to_decimal_above_top(self):
        self.assertRaises(Exception, self.cvt.convert_to_decimal, "MMMMI")

    
    def test_to_roman_tier1(self):
        self.assertEquals("V", self.cvt.convert_to_roman(5))

    def test_to_roman_tier2(self):
        self.assertEquals("X", self.cvt.convert_to_roman(10))

    def test_to_roman_tier3(self):
        self.assertEquals("L", self.cvt.convert_to_roman(50))

    def test_to_rman_tier4(self):
        self.assertEquals("C", self.cvt.convert_to_roman(100))

    def test_to_roman_tier5(self):
        self.assertEquals("D", self.cvt.convert_to_roman(500))

    def test_to_roman_tier6(self):
        self.assertEquals("M", self.cvt.convert_to_roman(1000))



    def test_to_roman_bad_inputs(self):
        self.assertEquals("", self.cvt.convert_to_roman(None))
        self.assertEquals("I", self.cvt.convert_to_roman(1.2))

    def test_to_decimal_bad_inputs(self):
        self.assertRaises(TypeError, self.cvt.convert_to_decimal, None)
        self.assertRaises(TypeError, self.cvt.convert_to_decimal, 1.2)    


if __name__ == '__main__':
    unittest.main()
