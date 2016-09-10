import os
BASE_DIR = os.path.dirname(__file__)

import pygame


class Menu():

    def __init__(self, screen):
        self.screen = screen
        self.brush = None

        #  colors
        self.colors = [ \
                        (0xff, 0x00, 0xff), (0x80, 0x00, 0x80), \
                        (0x00, 0x00, 0xff), (0x00, 0x00, 0x80), \
                        (0x00, 0xff, 0xff), (0x00, 0x80, 0x80), \
                        (0x00, 0xff, 0x00), (0x00, 0x80, 0x00), \
                        (0xff, 0xff, 0x00), (0x80, 0x80, 0x00), \
                        (0xff, 0x00, 0x00), (0x80, 0x00, 0x00), \
                        (0xc0, 0xc0, 0xc0), (0xff, 0xff, 0xff), \
                        (0x00, 0x00, 0x00), (0x80, 0x80, 0x80), \
                      ]
        self.colors_rect = []
        for (i, rgb) in enumerate(self.colors):
            rect = pygame.Rect(10 + i%2*32, 254 + i/2*32, 32, 32)
            self.colors_rect.append(rect)

        #  pens
        img_pen1 = os.path.join(BASE_DIR, 'resources/images/pen1.png')
        img_pen2 = os.path.join(BASE_DIR, 'resources/images/pen2.png')
        self.pens = [pygame.image.load(img_pen1).convert_alpha(), \
                     pygame.image.load(img_pen2).convert_alpha()]  
        self.pens_rect = []
        for (i, img) in enumerate(self.pens):
            rect = pygame.Rect(10, 10 + i*64, 64, 64)
            self.pens_rect.append(rect)

        #  sizes
        img_big = os.path.join(BASE_DIR, 'resources/images/big.png')
        img_small = os.path.join(BASE_DIR, 'resources/images/small.png')
        self.sizes = [pygame.image.load(img_big).convert_alpha(), \
                      pygame.image.load(img_small).convert_alpha()]
        self.sizes_rect = []
        for (i, img) in enumerate(self.sizes):
            rect = pygame.Rect(10 + i*32, 138, 32, 32)
            self.sizes_rect.append(rect)
 
    def set_brush(self, brush):
        self.brush = brush

    def draw(self):

        #  draw pen style button
        for (i, img) in enumerate(self.pens):
            self.screen.blit(img, self.pens_rect[i].topleft)

        #  draw <> buttons
        for (i, img) in enumerate(self.sizes):
            self.screen.blit(img, self.sizes_rect[i].topleft)

        #  draw current pen / color
        self.screen.fill((255, 255, 255), (10, 180, 64, 64))
        pygame.draw.rect(self.screen, (0, 0, 0), (10, 180, 64, 64), 1)
        
        size = self.brush.get_size()
        x = 10 + 32
        y = 180 + 32

        if self.brush.get_brush_style():
            x = x - size
            y = y - size
            self.screen.blit(self.brush.get_current_brush(), (x, y))
        else:
            pygame.draw.circle(self.screen, \
                               self.brush.get_color(), \
                               (x, y), \
                               size)


        #  draw colors panel
        for (i, rgb) in enumerate(self.colors):
            pygame.draw.rect(self.screen, rgb, self.colors_rect[i])

    def click_button(self, pos):
        #  pen buttons
        for (i, rect) in enumerate(self.pens_rect):
            if rect.collidepoint(pos):
                self.brush.set_brush_style(bool(i))
                return True

        #  size buttons
        for (i, rect) in enumerate(self.sizes_rect):
            if rect.collidepoint(pos):
                if i:  # i == 1, size down
                    self.brush.set_size(self.brush.get_size() - 0.5)
                else:
                    self.brush.set_size(self.brush.get_size() + 0.5)
                return True

        #  color buttons
        for (i, rect) in enumerate(self.colors_rect):
            if rect.collidepoint(pos):
                self.brush.set_color(self.colors[i])
                return True

        return False
