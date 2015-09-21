"""
Author: Kelly Chan
Date: July 27 2014
"""

import pygame
from pygame.locals import *

from util import myprint
from sound import play_sound


class Menu:

    OPTS = ['LEVEL 1', \
            'LEVEL 2', \
            'LEVEL 3', \
            'LEVEL 4', \
            'LEVEL 5', \
            'LEVEL 6', \
            'QUIT']

    def __init__(self, screen):
        self.screen = screen
        self.current = 0

    def run(self, elapse):
        self.draw()
        for e in pygame.event.get():
            if e.type == QUIT:
                return r'quit'
            elif e.type == KEYDOWN:
                if e.key == K_UP:
                    self.current = (self.current - 1) % len(self.OPTS)
                    play_sound('menu')
                elif e.key == K_DOWN:
                    self.current = (self.current + 1) % len(self.OPTS)
                    play_sound('menu')
                elif e.key == K_RETURN:
                    return self.OPTS[self.current].lower()
        return 'menu'
    
    def draw(self):
        for idx in xrange(len(self.OPTS)):
            if idx == self.current:
                myprint(self.screen, self.OPTS[idx], (200, 50*idx+50))
            else:
                myprint(self.screen, self.OPTS[idx], (200, 50*idx+50), 's', (160,160,160))

