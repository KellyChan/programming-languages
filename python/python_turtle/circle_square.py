"""
Project: Circling squares
Author: Kelly Chan
Date: Apr 30 2014
"""

import turtle

def drawCircleFromSquares():

    ang = turtle.Turtle()
    ang.shape("turtle")
    ang.color("white")
    ang.speed(20)

    repeat = 60
    for i in range(repeat):
        for i in range(4):
            ang.forward(100)
            ang.right(90)
        ang.rt(360/repeat)

def main():
    window = turtle.Screen()
    window.bgcolor("black")

    drawCircleFromSquares()

    window.exitonclick()


if __name__ == '__main__':
    main()

