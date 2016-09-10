"""
Author: Kelly Chan
Date: July 27 2014
"""

#!/usr/bin/env python
# -*-coding : utf-8 -*-

import pygame
from util import file_path

_SOUNDS = {}


def load():
    global _SOUNDS
    pygame.mixer.music.load(file_path('background.ogg'))
    pygame.mixer.music.play(-1)
    for event in ('1', '2', '3', '4', 'drop', 'menu'):
        _SOUNDS[event] = pygame.mixer.Sound(file_path(event + '.ogg'))

def play_sound(name):
    try:
        _SOUNDS[name].play()
    except KeyError:
        pass


