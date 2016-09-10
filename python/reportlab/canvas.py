import settings
import generator

from reportlab.pdfgen import canvas

def hello(c):
    c.drawString(100, 100, "Hello")



def main():

    c = canvas.Canvas("hello.pdf")
    hello(c)
    c.showPage()
    c.save()

if __name__ == '__main__':
    main()
