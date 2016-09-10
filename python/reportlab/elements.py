from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch


def hello(c):

    # move the origin up and to the left
    c.translate(inch, inch)

    # define a large font
    c.setFont("Helvetica", 14)

    # choose some colors
    c.setStrokeColorRGB(0.2, 0.5, 0.3)
    c.setFillColorRGB(1, 0, 1)

    # draw some lines
    c.line(0, 0, 0, 1.7*inch)
    c.line(0, 0, 1*inch, 0)

    # draw a rectange
    c.rect(0.2*inch, 0.2*inch, 1*inch, 1.5*inch, fill=1)

    # make text go straight up
    c.rotate(90)

    # change color
    c.setFillColorRGB(0, 0, 0.77)

    # print text (note after rotate the y coord needs to be negative!)
    c.drawString(0.3*inch, -inch, "Test")




def main():

    c = canvas.Canvas("outputs/elements.pdf", pagesize=letter)
    width, height = letter

    hello(c)
    c.showPage()
    c.save()

if __name__ == '__main__':
    main()
