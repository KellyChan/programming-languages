#!/usr/bin/env python

import pygame as pg

def main():

    sources = "G:/vimFiles/python/projects/piano-keybroad/sources/"
    kb_file = open(sources + "my_keyboard.kb", 'w')

    pg.init()

    screen = pg.display.set_mode((400, 400))
    print ("Press the keys in the right order. Press Escape to finish.")

    while True:

        event = pg.event.wait()
        
        if event.type is pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                break
            else:
                name = pg.key.name(event.key)
                print ("Last key pressed: %s" % name)
                kb_file.write(name + '\n')

    kb_file.close()
    print ("Done. You have a new keyboard configuration file: %s" % (kb_file))
    pg.quit()


if __name__ == '__main__':
    main()


