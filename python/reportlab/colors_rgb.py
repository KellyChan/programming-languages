from reportlab.pdfgen import canvas

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch


def colors_rgb(c):

    black = colors.black

    y = x = 0
    dy = inch * 3 / 4.0
    dx = inch * 5.5 / 5
    w = h = dy / 2;
    rdx = (dx - w) / 2
    rdy = h / 5.0
    texty = h + 2 * rdy

    colors_list = (
                   [colors.lavenderblush, "lavenderblush"], \
                   [colors.lawngreen, "lawngreen"], \
                   [colors.lemonchiffon, "lemonchiffon"], \
                   [colors.lightblue, "lightblue"], \
                   [colors.lightcoral, "lightcoral"] \
                  )

    rgb_list = [
                 (1, 0, 0), \
                 (0, 1, 0), \
                 (0, 0, 1), \
                 (0.5, 0.3, 0.1), \
                 (0.4, 0.5, 0.3) \
               ]

    c.setFont("Helvetica", 10)

    for [color_value, color_name] in colors_list:
        c.setFillColor(color_value)
        c.rect(x+rdx, y+rdy, w, h, fill=1)
        c.setFillColor(black)
        c.drawCentredString(x+dx/2, y+texty, color_name)
        x = x + dx

    y = y + dy
    x = 0

    for rgb in rgb_list:
        r, g, b = rgb
        c.setFillColorRGB(r, g, b)
        c.rect(x+rdx, y+rdy, w, h, fill=1)
        c.setFillColor(black)
        c.drawCentredString(x+dx/2, y+texty, "r%s g%s b%s" % rgb)
        x = x + dx

    y = y + dy
    x = 0

    for gray in (0.0, 0.25, 0.50, 0.75, 1.0):
        c.setFillGray(gray)
        c.rect(x+rdx, y+rdy, w, h, fill=1)
        c.setFillColor(black)
        c.drawCentredString(x+dx/2, y+texty, "gray: %s" % gray)
        x = x + dx

def init_canvas(outpath):

    c = canvas.Canvas(outpath, pagesize=letter)
    width, height = letter
    return c

def create_canvas(c):
    c.showPage()
    c.save()


def main():


    colors_dir = "outputs/colors.pdf"

    c_colors = init_canvas(colors_dir)
    colors_rgb(c_colors)
    create_canvas(c_colors)


if __name__ == '__main__':
    main()
