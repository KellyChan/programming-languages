#!/usr/bin/env python
# -*-coding : utf-8 -*-

import tetris

class Main:
    
    def __init__(self, screen):
        self.screen = screen

    def run(self, elapse):
        return self.tetris.update(elapse)

    def start(self, level):
        if level == 6:
            self.tetris = tetris.Tetris(self.screen)
        else:
            self.tetris = eval("tetris.Tetris"+str(level)+"(self.screen)")
