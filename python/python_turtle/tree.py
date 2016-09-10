"""
Project: tree
Author: Kelly Chan
Date: Apr 30 2014
"""

import turtle
import random

def drawFractalTree(aTurtle, depth, maxdepth):

    if depth > maxdepth:
        return
    else:
        for i in range(2):
            rand = random.randrange(-30, 30)
            aTurtle.left(rand)
            aTurtle.forward(50*(0.8)**depth)
            anotherTurtle = aTurtle.clone()
            drawFractalTree(anotherTurtle, depth+1, maxdepth)
        return



def drawPic():

    window = turtle.Screen()
    window.bgcolor("black")

    brad = turtle.Turtle()
    brad.shape("turtle")
    brad.color("white")

    brad.penup()
    brad.goto(0, -200)
    brad.left(90)

    brad.pendown()
    brad.hideturtle()
    brad.speed(0)
    
    drawFractalTree(brad, 0, 12)
    window.exitonclick()

if __name__ == '__main__':
    drawPic()
