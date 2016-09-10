#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import util
from sound import play_sound
from shape import Tile, Shape

class Tetris(object):
    W = 12              #width of board
    H = 20              #height of board
    TILEW = 20          #pixel width/height of a tile
    START = (100, 40)   #board lefttop on background
    SPACE = 1000        #delay of level xxx ms

    def __init__(self, screen):
        self.stat = "game"
        self.WIDTH = self.TILEW * self.W
        self.HEIGHT = self.TILEW * self.H
        self.screen = screen
        self.pause = False
        self.board = [] #save current situation same as screen
        for i in xrange(self.H):
            line = [None] * self.W
            self.board.append(line)
        #next to display 
        self.level = 1
        self.killed = 0
        self.score = 0
        #delay for shape fall
        self.time = self.SPACE * 0.8 ** (self.level - 1)
        #save the elapsed time after last fail
        self.elapsed = 0
        #pressd firstly time
        self.pressing = 0
        #moving shape
        self.shape = Shape(self.START,      \
                (self.WIDTH, self.HEIGHT), (self.W, self.H))
        self.shape.set_board(self.board)
        self.board_image = pygame.Surface((self.WIDTH, self.HEIGHT))
        #draw background
        self.screen.blit(pygame.image.load( \
                util.file_path("background.jpg")).convert(), (0, 0))
        self.display_info()

    def update(self, elapse):
        for e in pygame.event.get():
            if e.type == KEYDOWN:
                self.pressing = 1
                self.move(e.key == K_UP, e.key ==K_DOWN,    \
                          e.key == K_LEFT, e.key == K_RIGHT)
                if e.key == K_ESCAPE:
                    self.stat = 'menu'
                elif e.key == K_SPACE:
                    self.pause = not self.pause
            elif e.type == KEYUP and self.pressing:
                self.pressing = 0
            elif e.type == QUIT:
                self.stat = 'quit'
        if self.pause:
            self.draw()
            return self.stat
        if self.pressing:
            pressed = pygame.key.get_pressed()
            self.move(pressed[K_UP], pressed[K_DOWN],   \
                      pressed[K_LEFT], pressed[K_RIGHT])
        self.elapsed += elapse
        if self.elapsed >= self.time:
            self.next()
            self.elapsed = self.elapsed - self.time
            self.draw()
        return self.stat

    def move(self, u, d, l, r):
        if not(u or d or l or r):
            return
        #pressed for the first time or a long time
        if self.pressing == 1 or self.pressing > 10:
            if u:   self.shape.rotate()
            elif d: self.next()
            elif l: self.shape.move(-1, 0)
            elif r: self.shape.move(1, 0)
            self.draw()
        self.pressing += 1

    def check_line(self):
        self.add_to_board()
        filled = []
        for i in xrange(self.H-1, -1, -1):
            line = self.board[i]
            sum = 0
            for t in line:
                sum += 1 if t else 0
            if sum == self.W:
                filled.append(i)
            elif sum == 0:
                break
        if i == 0 and sum !=0:
            self.game_over()
        self.create_board_image()   #killing animation
        self.kill_line(filled)
        self.create_board_image()   #update

    def kill_line(self, filled = []):
        if len(filled)  == 0:
            return
        play_sound(str(len(filled)))
        mask = pygame.Surface((self.WIDTH, self.TILEW), SRCALPHA, 32)
        for i in xrange(5):
            if i % 2 == 0:
                mask.fill((255, 255, 255, 100))
            else:
                mask.fill((255, 255, 255, 200))
            self.screen.blit(self.board_image, self.START)
            for line in filled:
                self.screen.blit(mask,      \
                    (self.START[0], self.START[1]+line*self.TILEW))
                pygame.display.update()
            pygame.time.wait(80)
        #remove filled lines
        [self.board.pop(l) for l in sorted(filled, reverse=True)]
        #fill with blank lines
        [self.board.insert(0, [None]*self.W) for l in filled]
        self.get_score(len(filled))

    def get_score(self, num):
        self.killed += num
        self.score += num * num * 10 * self.level
        self.level = 1 + self.killed // 10
        self.time = self.SPACE * 0.8 ** (self.level - 1)

    def add_to_board(self):
        for x in xrange(self.shape.SHAPE_WIDTH):
            for y in xrange(self.shape.SHAPE_HEIGHT):
                if self.shape.shape[y][x]:
                    self.board[self.shape.y+y][self.shape.x+x] = Tile(  \
                                self.shape.color)

    def create_board_image(self):
        self.board_image.fill((0, 0, 0))
        for x in xrange(self.H):
            for y in xrange(self.W):
                if self.board[x][y]:
                    rect = pygame.Rect(y * self.TILEW, x * self.TILEW,   \
                            self.TILEW, self.TILEW)
                    self.board_image.fill(self.board[x][y].color, rect)
                    pygame.draw.rect(self.board_image,      \
                            (200, 200, 200, 100), rect, 1)

    def next(self):
        if self.shape.at_bottom():
            play_sound('drop')
            self.check_line()
            self.shape.new()
            self.display_info()
        else:
            self.shape.move(0, 1)

    def draw(self):
        self.screen.blit(self.board_image, self.START)
        self.shape.draw(self.screen)
        if self.pause:
            util.myprint(self.screen, "PAUSE",      \
                    (self.START[0]+50, self.START[1]+200), "m")

    def display_info(self):
        self._display_next()
        self._display_score()

    def _display_score(self):
        try:
            self._score_board
        except AttributeError:
            self._score_board = (
                                 self.START[0]+self.WIDTH+30,   \
                                 self.START[1]+100,             \
                                 200,
                                 260)
            self._score_level = (
                        self._score_board[0] + 10,
                        self._score_board[1] + 10)
            self._score_level_v = (
                        self._score_board[0] + 30,
                        self._score_board[1] + 50)
            self._score_killed = (
                        self._score_board[0] + 10,
                        self._score_board[1] + 90)
            self._score_killed_v = (
                        self._score_board[0] + 30,
                        self._score_board[1] + 130)
            self._score_score = (
                        self._score_board[0] + 10,
                        self._score_board[1] + 170)
            self._score_score_v = (
                        self._score_board[0] + 30,
                        self._score_board[1] + 210)
        self.screen.fill((0, 0, 0), self._score_board)
        util.myprint(self.screen, 'LEVEL', self._score_level, 'm')
        util.myprint(self.screen, self.level, self._score_level_v, 'm')
        util.myprint(self.screen, 'LINES', self._score_killed, 'm')
        util.myprint(self.screen, self.killed, self._score_killed_v, 'm')
        util.myprint(self.screen, 'SCORE', self._score_score, 'm')
        util.myprint(self.screen, self.score, self._score_score_v, 'm')

    def _display_next(self):
        try:
            self._next_board
        except AttributeError:
            self._next_board = (
                    self.START[0] + self.WIDTH + 30,
                    self.START[1],
                    200,
                    80)
            self._next_board_v = (
                    self.START[0] + self.WIDTH + 30 + 100 - self.TILEW *2,
                    self.START[1] + 40 - self.TILEW * 2)
        self.screen.fill((0, 0, 0), self._next_board)
        self.screen.blit(self.shape.image_next, self._next_board_v)

    def game_over(self):
        self.stat = "menu"

class Tetris1(Tetris):
    """ level 1, any block drop to bottom will be killed """
    def check_line(self):
        self.add_to_board()
        filled = []
        for i in xrange(self.H - 1, -1, -1):
            line = self.board[i]
            sum = 0
            for t in line:
                sum += 1 if t else 0
            if sum != 0:
                filled.append(i)
            elif sum == 0:
                break
        if i == 0 and sum != 0:
            self.game_over()
        self.create_board_image()
        self.kill_line(filled)
        self.create_board_image()

class Tetris2(Tetris):
    """ level 2, just have long stick falls """
    def __init__(self, screen):
        super(Tetris2, self).__init__(screen)
        from shape import Shape2
        self.shape = Shape2(self.START,         \
            (self.WIDTH, self.HEIGHT), (self.W, self.H))
        self.shape.set_board(self.board)

class Tetris3(Tetris):
    """ level 3, tile will be very small """
    W = 60
    H = 100
    TILEW = 4

class Tetris4(Tetris):
    """ level 4, cat model """
    def __inti__(self, screen):
        super(Tetris4, self).__init__(screen)
        from shape import Shape4
        self.shape = Shape4(self.START,         \
            (self.WIDTH, self.HEIGHT), (self.W, self.H))
        self.shape.set_board(self.board)

    def add_to_board(self):
        for x in xrange(self.shape.SHAPE_WIDTH):
            for y in xrange(self.shape.SHAPE_HEIGHT):
                if self.shape.shape[y][x]:
                    if self.shape.color == (0, 0, 0):
                        self.board[self.shape.y+y][self.shape.x+x] =    \
                                Tile(self.shape.color,                  \
                                     self.shape.get_part_image(y, x))
                    else:
                        self.board[self.shape.y+y][self.shape.x+x] =    \
                                Tile(self.shape.color)

    def create_board_image(self):
        self.board_image.fill((0, 0, 0))
        for x in xrange(self.H):
            for y in xrange(self.W):
                if self.board[x][y]:
                    rect = pygame.Rect(y * self.TILEW, x * self.TILEW,
                                self.TILEW, self.TILEW)
                    if self.board[x][y].color == (0, 0, 0):
                        self.board_image.blit(self.board[x][y].image, rect)
                    else:
                        self.board_image.fill(self.board[x][y].color, rect)
                    pygame.draw.rect(self.board_image,
                            (200, 200, 200, 100), rect, 1)

class Tetris5(Tetris):
    pass  
