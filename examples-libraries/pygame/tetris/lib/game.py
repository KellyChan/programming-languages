#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
This is the game main loop 
"""

import pygame
from pygame.locals import *
from menu import Menu
from main import Main
import sound, util

class Game:
    """
    This is the game manager modle 
    """
    def __init__(self):
        self.stat = 'menu'
        pygame.mixer.pre_init(44100, 16, 2, 1024*4) #mixer pre_init arguments
        pygame.init()
        pygame.display.set_caption("TETRIS_FUNNY") 
        self.init()
        try:
            self.screen = pygame.display.set_mode((640, 480),
                    HWSURFACE | SRCALPHA, 32)
        except:
            self.screen = pygame.display.set_mode((640, 480),
                    SRCALPHA, 32)
        try:
            pygame.display.set_icon(pygame.image.load(
                        util.file_path("icon.png")).convert_alpha())
        except:
            print "can't find icon picture under data directory"
            pass
        #init sub modules
        self.menu = Menu(self.screen)
        self.main = Main(self.screen)
    
    def init(self):
        util.init()
        sound.load()

    def loop(self):
        clock = pygame.time.Clock()
        while self.stat != r'quit':
            elapse = clock.tick(60)
            if self.stat == r'menu':
                self.stat = self.menu.run(elapse)
            elif self.stat == r'game':
                self.stat = self.main.run(elapse)

            if self.stat.startswith('level'):
                level = int(self.stat.split()[1])
                print "Start game at level", level
                self.main.start(level)
                self.stat = r"game"
            pygame.display.update()
        pygame.quit()

def run():
    tetris = Game()
    tetris.loop()

if __name__ == "__main__":
    print "Please run the game with 'run_game.py!'"

 
