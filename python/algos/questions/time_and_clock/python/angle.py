"""
Given a time, calculate the angle between the hour and minute hands.
"""

import doctest


def calculate_time_angle(time):
    """
    Returns the angle between the hour and minute hands by time.

    >>> calculate_time_angle("12:33")
    178.5

    >>> calculate_time_angle("23:23")
    203.5

    >>> calculate_time_angle("-1")
    'TimeFormatError: (i.e. 10:00)'
    """

    try:
        hour, minute = time.split(":")
        return (30 * (int(hour) % 12) - 5.5 * int(minute)) % 360
    except:
        return "TimeFormatError: (i.e. 10:00)"


if __name__ == '__main__':
    doctest.testmod()
