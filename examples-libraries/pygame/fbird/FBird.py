#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Kelly Chan
# Date: Aug 4 2014


import os
import sys
from sys import exit

from random import *

import pygame
from pygame.locals import *
from gameobjects.vector2 import Vector2


background_image = r'background.png'
field_image = r'field.png'
pipe1_image = r'pipe1.png'
pipe2_image = r'pipe2.png'
bird_image = r'bird1.png'

SCORE = 0
SCREEN_SIZE = (240, 460)

def load_image(name, colorkey = None):

    fullname = os.path.join("data", "image", name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error, message:
        print 'Cannot load image: ', fullname
        raise SystemExit, message

    image = image.convert_alpha()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at(0, 0)
        image.set_colorkey(colorkey, RLEACCEL)

    return image


def game_reinit():
    while True:
        event = pygame.event.wait()
        if event.type == QUIT:
            exit()
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                return True


class Pipe(pygame.sprite.Sprite):

    def __init__(self, image_file, position, num):
        pygame.sprite.Sprite.__init__(self)
        self.speed = 120
        self.pipe_mid = 440
        self.direction = Vector2(-1, 0)
        self.image = load_image(image_file)
        self.rect = self.image.get_rect()
        self.rect.topleft = position
        self.number = num

    def update(self, time_passed_seconds, pipe_down):
        pipe_move = self.direction * self.speed * time_passed_seconds
        self.rect = self.rect.move(pipe_move)
        if self.rect.left < -40:
            self.rect.left = 240
            temp = self.rect.top + pipe_down
            
            if self.number is 1:
                if temp < -110 or temp > -170:
                    self.rect.top = -140 + pipe_down
                else:
                    self.rect.top = -140

            if self.number is 2:
                if temp > 230 or temp < 290:
                    self.rect.top = 260 + pipe_down
                else:
                    self.rect.top = 260


class Bird(pygame.sprite.Sprite):

    def __init__(self):
            pygame.sprite.Sprite.__init__(self)
            self.image = load_image(bird_image)
            self.rect = self.image.get_rect()
            self.rect.topleft = Vector2(120.0, 180.0)
            self.direction = 0
            self.speed = 400
            self.gravity = 3.5
            self.alive = True

    def update(self, time_passed_seconds):
        if self.alive:
            self.rect.top += self.direction * self.speed * time_passed_seconds + self.gravity


class Field(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = load_image(field_image)
        self.rect = self.image.get_rect()
        self.rect.topleft = Vector2(0, 380)

    def update(self, cls):
        pass


def main():

    global SCORE
    
    #  initialize
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32)
    pygame.display.set_caption("FBird")
    my_font = pygame.font.SysFont("arial", 16)

    #  create and display background
    background = load_image(background_image)
    screen.blit(background, (0, 0))

    #  set clock
    clock = pygame.time.Clock()
    sum_time = 0

    #  create objects
    bird = Bird()
    field = Field()
    
    mypipe = [n for n in range(4)]
    mypipe[0] = Pipe(pipe1_image, Vector2(200, -140), 1)
    mypipe[1] = Pipe(pipe2_image, Vector2(200, 260), 2)
    mypipe[2] = Pipe(pipe1_image, Vector2(60, -140), 1)
    mypipe[3] = Pipe(pipe2_image, Vector2(60, 260), 2)

    #  group objects
    pipe_group = pygame.sprite.RenderPlain(*mypipe)
    allsprites = pygame.sprite.RenderPlain(bird, *mypipe)

    #  main loop
    while True:
        clock.tick(40)
        screen.blit(background, (0, 0))
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        pressed_keys = pygame.key.get_pressed()
        bird.direction = 0
        if pressed_keys[K_UP]:
            bird.direction = -1

        time_passed = clock.get_time()
        time_passed_seconds = time_passed / 1000.0
        sum_time += time_passed

        #  when the bird touches the field, failed
        if pygame.sprite.collide_rect(bird, field) == True:
            bird.alive = False
        #  when the bird touches the pipes, failed
        for i in xrange(4):
            if pygame.sprite.collide_rect(bird, mypipe[i]) == True:
                bird.alive = False

        bird.update(time_passed_seconds)
        pipe_group.update(time_passed_seconds, randint(-50, 50))
        allsprites.draw(screen)
        screen.blit(field.image, field.rect)
        
        if bird.alive:
            SCORE = sum_time / 1000
            score_surface = my_font.render("Score:" + str(SCORE), True, (0,0,0))
            screen.blit(score_surface, (90, 400))
        else:
            if game_reinit():
                main()

        pygame.display.flip()


if __name__ == '__main__':
    main()


