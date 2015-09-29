from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import pink, black, red, blue, green

def _coords(c):

    c.setStrokeColor(pink)
    c.grid([inch, 2*inch, 3*inch, 4*inch], [0.5*inch, inch, 1.5*inch, 2*inch, 2.5*inch])

    c.setStrokeColor(black)
    c.setFont("Times-Roman", 20)
    c.drawString(0, 0, "(0, 0) the Origin")
    c.drawString(2.5*inch, inch, "(2.5, 1) in inches")
    c.drawString(4*inch, 2.5*inch, "4, 2.5")
    c.setFillColor(red)

def translate(c):
    c.translate(2.3*cm, 0.3*cm)
    _coords(c)

def scale(c):

    c.scale(0.75, 0.5)
    _coords(c)


def init_canvas(outpath):

    c = canvas.Canvas(outpath, pagesize=letter)
    width, height = letter
    return c

def create_canvas(c):
    c.showPage()
    c.save()

def main():


    origin_dir = "outputs/coords-origin.pdf"
    translate_dir = "outputs/coords-translate.pdf"
    scale_dir = "outputs/coords-scale.pdf"

    c_origin = init_canvas(origin_dir)
    _coords(c_origin)
    create_canvas(c_origin)

    c_translate = init_canvas(translate_dir)
    scale(c_translate)
    create_canvas(c_translate)

    c_scale = init_canvas(scale_dir)
    scale(c_scale)
    create_canvas(c_scale)

if __name__ == '__main__':
    main()
