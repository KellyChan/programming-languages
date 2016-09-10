import unittest

import pieces.general

class GeneralTests(unittest.TestCase):

    def setUp(self):
        self.general = pieces.general.General()

    def test_possible_moves(self):
        self.assertEquals('', self.general.possible_moves())
