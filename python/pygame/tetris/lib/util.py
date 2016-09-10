"""
Author: Kelly Chan
Date: July 27 2014
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pygame
from pygame.locals import *

if hasattr(sys, 'frozen'):  #  cx_freeze used for make exe for win platform
    _ME_PATH = os.path.abspath(os.path.dirname(sys.executable))
    DATA_PATH = os.path.normpath(os.path.join(_ME_PATH, 'data'))
else:
    _ME_PATH = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH = os.path.normpath(os.path.join(_ME_PATH, '..', 'data'))

_FONTS = {}


def file_path(filename=None):
    """ auto add fullname for the supplied file """
    
    if filename is None:
        raise ValueError, "must supply a filename"

    fileext = os.path.splitext(filename)[1]

    if fileext in('.png', '.bmp', '.tga', '.jpg'):
        sub = 'images'
    elif fileext in ('.ogg', '.mp3', '.wav'):
        sub = 'sounds'
    elif fileext in ('.ttf'):
        sub = 'fonts'

    file_path = os.path.join(DATA_PATH, sub, filename)
    #print 'will read', file_path

    try:
        f_path = os.path.abspath(file_path)
        return f_path
    except ValueError:
        raise "It can not open file: '%s'" % file_path


def myprint(screen, string, pos, size='s', color=(255, 255, 255)):
    
    fs = _FONTS[size].render(str(string), True, color)
    screen.blit(fs, pos)

def init():
    print "DATA PATH is: ", DATA_PATH
    global _FONTS
    _FONTS['l'] = pygame.font.Font(file_path('default.ttf'), 48)
    _FONTS['m'] = pygame.font.Font(file_path('default.ttf'), 36)
    _FONTS['s'] = pygame.font.Font(file_path('default.ttf'), 24)



