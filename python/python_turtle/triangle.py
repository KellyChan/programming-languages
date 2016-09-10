"""
Project: Triangle
Author: Kelly Chan
Date: Apr 30 2014
"""

import turtle

def drawFractalTriangle(aTurtle, depth, maxdepth):

    if depth > maxdepth:
        return
    else:
        for i in range(3):
            aTurtle.forward(256/2**depth)
            drawFractalTriangle(aTurtle, depth+1, maxdepth)
            aTurtle.forward(256/(2**depth))
            aTurtle.left(120)
        return

def draw_pic():

    window = turtle.Screen()
    window.bgcolor("black")

    brad = turtle.Turtle()
    brad.shape("turtle")
    brad.color("white")

    brad.penup()
    brad.goto(-50, -200)

    brad.pendown()
    brad.hideturtle()
    brad.speed(0)

    brad.right(180)
    brad.forward(256)
    brad.right(180)
    
    drawFractalTriangle(brad, 0, 8)

    window.exitonclick()

if __name__ == '__main__':

    draw_pic()


