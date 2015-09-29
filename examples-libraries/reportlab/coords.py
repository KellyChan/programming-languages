from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.colors import pink, black, red, blue, green

def coords(c):

    c.setStrokeColor(pink)
    c.grid([inch, 2*inch, 3*inch, 4*inch], [0.5*inch, inch, 1.5*inch, 2*inch, 2.5*inch])

    c.setStrokeColor(black)
    c.setFont("Times-Roman", 20)
    c.drawString(0, 0, "(0, 0) the Origin")
    c.drawString(2.5*inch, inch, "(2.5, 1) in inches")
    c.drawString(4*inch, 2.5*inch, "4, 2.5")
    c.setFillColor(red)


def main():

    c = canvas.Canvas("outputs/coords.pdf", pagesize=letter)
    width, height = letter

    coords(c)
    c.showPage()
    c.save()

if __name__ == '__main__':
    main()
