import unittest
import isbalanced

class IsBalancedTests(unittest.TestCase):

    def test_balanced(self):
        self.assertEqual(isbalanced.isbalanced('{{[()]}}'), True)
        self.assertEqual(isbalanced.isbalanced('{[[()'), False)
        self.assertEqual(isbalanced.isbalanced('{dfd{[(dd)]}}'), True)
        self.assertEqual(isbalanced.isbalanced('{[[(dd)'), False)
