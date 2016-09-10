"""
Author: Kelly Chan
Date: July 25 2014
"""

import os
import sys
import math
import pygame
from pygame.locals import *

import brush
import menu

 
class Painter():

    def __init__(self):
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Painter")
        self.clock = pygame.time.Clock()
        self.brush = brush.Brush(self.screen)
        self.menu  = menu.Menu(self.screen)
        self.menu.set_brush(self.brush)
 
    def run(self):
        self.screen.fill((255, 255, 255))
        while True:
            # max fps limit
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                elif event.type == KEYDOWN:
                    # press esc to clear screen
                    if event.key == K_ESCAPE:
                        self.screen.fill((255, 255, 255))
                elif event.type == MOUSEBUTTONDOWN:
                    # <= 74, coarse judge here can save much time
                    if ((event.pos)[0] <= 74 and
                            self.menu.click_button(event.pos)):
                        # if not click on a functional button, do drawing
                        pass
                    else:
                        self.brush.start_draw(event.pos)
                elif event.type == MOUSEMOTION:
                    self.brush.draw(event.pos)
                elif event.type == MOUSEBUTTONUP:
                    self.brush.end_draw()
 
            self.menu.draw()
            pygame.display.update()
 
if __name__ == '__main__':
    app = Painter()
    app.run()
