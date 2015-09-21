#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import pygame
from pygame.locals import *

import util

class Tile:
    def __init__(self, color, image = None):
        self.color = color
        self.image = image

class Shape(object):

    SHAPE_WIDTH = 4
    SHAPE_HEIGHT = 4
    
    SHAPES = (
        (   ((0,0,0,0),     #
             (0,1,1,0),     #   [][]
             (0,1,1,0),     #   [][]
             (0,0,0,0),),   #
        ),
        (   ((0,0,0,0),     #
             (1,1,1,1),     # [][][][]
             (0,0,0,0),     #
             (0,0,0,0),),   #
            ((0,1,0,0),     #   []
             (0,1,0,0),     #   []
             (0,1,0,0),     #   []
             (0,1,0,0),),   #   []
        ),
        (   ((0,0,0,0),     #
             (0,1,1,0),     #   [][]
             (1,1,0,0),     # [][]
             (0,0,0,0),),   #
            ((1,0,0,0),     # []
             (1,1,0,0),     # [][]
             (0,1,0,0),     #   []
             (0,0,0,0),),
        ),
        (   ((0,0,0,0),     #
             (1,1,0,0),     # [][]
             (0,1,1,0),     #   [][]
             (0,0,0,0),),   #
            ((0,1,0,0),     #   []
             (1,1,0,0),     # [][]
             (1,0,0,0),     # []
             (0,0,0,0),),   #
        ),
        (   ((0,0,0,0),     #
             (1,1,1,0),     # [][][]
             (1,0,0,0),     # []
             (0,0,0,0),),   #
            ((0,1,0,0),     #   []
             (0,1,0,0),     #   []
             (0,1,1,0),     #   [][]
             (0,0,0,0),),   #
            ((0,0,1,0),     #     []
             (1,1,1,0),     # [][][]
             (0,0,0,0),     #
             (0,0,0,0),),   #
            ((1,1,0,0),     # [][]
             (0,1,0,0),     #   []
             (0,1,0,0),     #   []
             (0,0,0,0),),   #
        ),
        (   ((0,0,0,0),     #
             (1,1,1,0),     # [][][]
             (0,0,1,0),     #     []
             (0,0,0,0),),   #
            ((0,1,1,0),     #   [][]
             (0,1,0,0),     #   []
             (0,1,0,0),     #   []
             (0,0,0,0),),   #
            ((1,0,0,0),     # []
             (1,1,1,0),     # [][][]
             (0,0,0,0),     #
             (0,0,0,0),),   #
            ((0,1,0,0),     #   []
             (0,1,0,0),     #   []
             (1,1,0,0),     # [][]
             (0,0,0,0),),   #
        ),
        (   ((0,0,0,0),     #
             (1,1,1,0),     # [][][]
             (0,1,0,0),     #   []
             (0,0,0,0),),   #
            ((0,1,0,0),     #   []
             (0,1,1,0),     #   [][]
             (0,1,0,0),     #   []
             (0,0,0,0),),   #
            ((0,1,0,0),     #   []
             (1,1,1,0),     # [][][]
             (0,0,0,0),     #
             (0,0,0,0),),   #
            ((0,1,0,0),     #   []
             (1,1,0,0),     # [][]
             (0,1,0,0),     #   []
             (0,0,0,0),),   #
        ),
    )

    COLORS = ((0xcc, 0x66, 0x66),(0x66, 0xcc, 0x66),    \
              (0x66, 0x66, 0xcc),(0xcc, 0xcc, 0x66),    \
              (0xcc, 0x66, 0xcc),(0x66, 0xcc, 0xcc),    \
              (0xda, 0xaa, 0x00))
    
    def __init__(self, board_start, (board_w, board_h), (w, h)):
        self.start = board_start
        self.W, self.H = w, h           #width of board
        self.length = board_w / w       #width/height of a title
        self.x = 0
        self.y = 0
        self.index = 0                  # the type of shape
        self.indexN = 0                 # the type of next shape
        self.subindex = 0               # the index in the specil shape
        self.shapes = []                # record the shapes
        self.color = ()
        self.shape = None
        self.image = pygame.Surface((self.length*self.SHAPE_WIDTH, \
                    self.length*self.SHAPE_HEIGHT), SRCALPHA, 32)
        self.image_next = pygame.Surface((self.length*self.SHAPE_WIDTH, \
                    self.length*self.SHAPE_HEIGHT), SRCALPHA, 32)
        self.board = []                 #the current board stat
        self.new()

    def set_board(self, board):
        self.board = board

    def new(self):
        self.x = self.W /2 - 2
        self.y = 0
        self.index = self.indexN
        self.shapes = self.SHAPES[self.index]
        self.subindex = random.randint(0, len(self.shapes) - 1)
        self.color = self.COLORS[self.index]
        self.shape = self.shapes[self.subindex]
        self.indexN = random.randint(0, len(self.SHAPES) - 1)
        self.draw_current_shape()
        self.draw_next_shape()

    def rotate(self):
        self.subindex = (self.subindex + 1) % len(self.shapes)
        self.shape = self.shapes[self.subindex]
        if self.check_legal():
            pass
        else:
            self.subindex = (self.subindex - 1) % len(self.shapes)
            self.shape = self.shapes[self.subindex]
        self.draw_current_shape()

    def check_legal(self, r=0, c=0):
        for x in xrange(self.SHAPE_WIDTH):
            for y in xrange(self.SHAPE_HEIGHT):
                if (self.shape[y][x] and            # a tile there
                   (self.x+x+r < 0 or               # left outside
                    self.x+x+r >= self.W or         # right outside
                    self.y+y+c >= self.H or         # bottom outside
                    self.board[self.y+y+c][self.x+x+r]  # tile cover
                    )):
                    return False
        return True

    def move(self, r, c):
        if self.check_legal(r, c):
            self.x += r
            self.y += c

    def at_bottom(self):
        for x in xrange(self.SHAPE_WIDTH):
            for y in xrange(self.SHAPE_HEIGHT - 1, -1, -1):
                if (self.shape[y][x] and        \
                    (self.y+y+1 >= self. H or   \
                     self.board[self.y+y+1][self.x+x])):
                    return True
        return False

    def draw_current_shape(self):
        self._draw_shape(self.image, self.index, self.subindex)

    def draw_next_shape(self):
        self._draw_shape(self.image_next, self.indexN)

    def _draw_shape(self, surface, index, subindex = -1):
        """ Draw the shape to surface """
        surface.fill((0, 0, 0, 0))
        if subindex == -1:
            subindex = 0
        shape = self.SHAPES[index][subindex]
        color = self.COLORS[index]
        for x in xrange(self.SHAPE_HEIGHT):
            for y in xrange(self.SHAPE_WIDTH):
                if shape[x][y]:
                    surface.fill(color,                 \
                        (y*self.length, x*self.length,  \
                        self.length, self.length))
                    pygame.draw.rect(surface,           \
                        (255, 255, 255, 100),           \
                        (y*self.length, x*self.length, self.length, self.length),   \
                        1)
    def draw(self, screen):
        screen.blit(self.image, (self.start[0]+self.length*self.x,      \
                                 self.start[1]+self.length*self.y))

class Shape2(Shape):
    def __init__(self, board_start, (board_width, board_height), (w, h)):
        self.SHAPES = ((((1,) * w, ), ), )
        self.SHAPE_WIDTH = w
        self.SHAPE_HEIGHT = 1
        super(Shape2, self).__init__(board_start,       \
                (board_width, board_height), (w, h))

    def new(self):
        self.x = 0
        self.y = 0
        self.index = self.indexN
        self.shapes = self.SHAPES[self.index]
        self.subindex = random.randint(0, len(self.shapes) - 1)
        self.color = self.COLORS[self.index]
        self.shape = self.shapes[self.subindex]
        self.indexN = random.randint(0, len(self.SHAPES) - 1)
        self.draw_current_shape()
        self.draw_next_shape()

    def draw_next_shape(self):
        pass

class Shape4(Shape):
    def __init__(self, board_start, (board_width, board_height), (w, h)):
        self.SHAPES += (
                (((1,1,1,1), (1,1,1,1), (1,1,1,1), (1,1,1,1)),),
    )
        self.COLORS += ((0, 0, 0),)
        self._image = {}
        self.image[7] = pygame.image.load(util.file_path('neko.png')).convert_alpha() 
        super(Shape4, self).__init__(board_start, (board_width, board_height), (w, h))

    def _draw_shape(self, surface, index, subindex = -1):
        surface.fill((0, 0, 0,0))
        if index > 6:
            surface.blit(self._image[index], (0, 0))
        else:
            super(Shape4, self)._draw_shape(surface, index, subindex)

    def get_part_image(self, x, y):
        return self._image[self.index].subsurface(      \
                (y*self.length, x*self.length), (self.length, self.length))
        

