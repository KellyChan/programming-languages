"""
Tests of angle.calculate_time_angle()
"""

import unittest

import angle


class TimeAngleTests(unittest.TestCase):
    """Tests of TimeAngle."""

    def test_extremes(self):
        """Tests extreme cases."""
        self.assertEquals(0., angle.calculate_time_angle("0:00"))
        self.assertEquals(0., angle.calculate_time_angle("12:00"))

    def test_user_errors(self):
        """Tests input errors."""
        self.assertEquals("TimeFormatError: (i.e. 10:00)",
                          angle.calculate_time_angle("10.00"))
        self.assertEquals("TimeFormatError: (i.e. 10:00)",
                          angle.calculate_time_angle("-1"))

    def test_general_cases(self):
        """Tests general cases."""
        self.assertEquals(143.0, angle.calculate_time_angle("23:34"))
