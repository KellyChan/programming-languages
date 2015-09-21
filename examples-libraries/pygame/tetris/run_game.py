"""
Author: Kelly Chan
Date: July 27 2014
"""

import os
import sys

try:
    LIBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
    sys.path.insert(0, LIBDIR)
except:
    sys.path.insert(0, os.path.join(sys.path[0], 'lib'))
    pass


import game


if __name__ == '__main__':
    game.run()


