import os
BASE_DIR = os.path.dirname(__file__)

import math
import pygame

class Brush():

    def __init__(self, screen):
        self.screen = screen
        self.color = (0, 0, 0)
        self.size = 1
        self.drawing = False
        self.last_pos = None
        self.space = 1
        #  style
        #  - True: normal solid brush
        #  - False: png brush
        self.style = False
        #  load brush style png
        img_brush = os.path.join(BASE_DIR, 'resources/images/brush.png')
        self.brush = pygame.image.load(img_brush).convert_alpha()
        #  set the current brush depends on size
        self.brush_now = self.brush.subsurface((0,0), (1,1))

    def start_draw(self, pos):
        self.drawing = True
        self.last_pos = pos

    def end_draw(self):
        self.drawing = False

    def set_brush_style(self, style):
        print "* set brush style to", style
        self.style = style

    def get_brush_style(self):
        return self.style

    def get_current_brush(self):
        return self.brush_now

    def set_size(self, size):
        if size < 1: size = 1 
        elif size > 32: size = 32
        print "* set brush size to", size
        self.size = size
        self.brush_now = self.brush.subsurface((0,0), (size*2, size*2))

    def get_size(self):
        return self.size

    def set_color(self, color):
        self.color = color
        for i in xrange(self.brush.get_width()):
            for j in xrange(self.brush.get_height()):
                self.brush.set_at((i, j), \
                                  color + (self.brush.get_at((i, j)).a,))

    def get_color(self):
        return self.color

    def draw(self, pos):
        if self.drawing:
            for p in self._get_points(pos):
                #  draw everypoint between them
                if self.style == False:
                    pygame.draw.circle(self.screen, self.color, p, self.size)
                else:
                    self.screen.blit(self.brush_now, p)

            self.last_pos = pos

    def _get_points(self, pos):
        """ get all points between last_point and now_point """

        points = [(self.last_pos[0], self.last_pos[1])]
        len_x = pos[0] - self.last_pos[0]
        len_y = pos[1] - self.last_pos[1]

        length = math.sqrt(len_x**2 + len_y**2)
        step_x = len_x / length
        step_y = len_y / length

        for i in xrange(int(length)):
            points.append((points[-1][0] + step_x, \
                           points[-1][1] + step_y))

        #  return light-weight, uniq integer point list
        return list(set(points))
 
