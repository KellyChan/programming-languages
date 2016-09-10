##########################
Reportlab
##########################

**********************
Canvas
**********************

import

::

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch


lines

::

    canvas.line(x1,y1,x2,y2)
    canvas.lines(linelist)

line styles

::

    canvas.setLineWidth(width)
    canvas.setLineCap(mode)
    canvas.setLineJoin(mode)
    canvas.setMiterLimit(limit)
    canvas.setDash(self, array=[], phase=0)


shapes

::

    canvas.grid(xlist, ylist)
    canvas.bezier(x1, y1, x2, y2, x3, y3, x4, y4)
    canvas.arc(x1,y1,x2,y2)
    canvas.rect(x, y, width, height, stroke=1, fill=0)
    canvas.ellipse(x1,y1, x2,y2, stroke=1, fill=0)
    canvas.wedge(x1,y1, x2,y2, startAng, extent, stroke=1, fill=0)
    canvas.circle(x_cen, y_cen, r, stroke=1, fill=0)
    canvas.roundRect(x, y, width, height, radius, stroke=1, fill=0)


string drawing

::

    canvas.drawString(x, y, text):
    canvas.drawRightString(x, y, text)
    canvas.drawCentredString(x, y, text)

text object

::

    textobject = canvas.beginText(x, y)
    canvas.drawText(textobject)


path object

::

    path = canvas.beginPath()
    canvas.drawPath(path, stroke=1, fill=0)
    canvas.clipPath(path, stroke=1, fill=0)


image

::

    canvas.drawInlineImage(self, image, x,y, width=None,height=None)
    canvas.drawImage(self, image, x,y, width=None,height=None,mask=None)

end page

::

    canvas.showPage()


colors

::

    canvas.setFillColorCMYK(c, m, y, k)
    canvas.setStrikeColorCMYK(c, m, y, k)
    canvas.setFillColorRGB(r, g, b)
    canvas.setStrokeColorRGB(r, g, b)
    canvas.setFillColor(acolor)
    canvas.setStrokeColor(acolor)
    canvas.setFillGray(gray)
    canvas.setStrokeGray(gray)

fonts

::

    canvas.setFont(psfontname, size, leading = None)

geometry

::

    canvas.setPageSize(pair)
    canvas.transform(a,b,c,d,e,f):
    canvas.translate(dx, dy)
    canvas.scale(x, y)
    canvas.rotate(theta)
    canvas.skew(alpha, beta)

state control

::

    canvas.saveState()
    canvas.restoreState()


others

::

    canvas.setAuthor()
    canvas.addOutlineEntry(title, key, level=0, closed=None)
    canvas.setTitle(title)
    canvas.setSubject(subj)
    canvas.pageHasData()
    canvas.showOutline()
    canvas.bookmarkPage(name)
    canvas.bookmarkHorizontalAbsolute(name, yhorizontal)
    canvas.doForm()
    canvas.beginForm(name, lowerx=0, lowery=0, upperx=None, uppery=None)
    canvas.endForm()
    canvas.linkAbsolute(contents, destinationname, Rect=None, addtopage=1, name=None, **kw)
    canvas.linkRect(contents, destinationname, Rect=None, addtopage=1, relative=1, name=None, **kw)
    canvas.getPageNumber()
    canvas.addLiteral()
    canvas.getAvailableFonts()
    canvas.stringWidth(self, text, fontName, fontSize, encoding=None)
    canvas.setPageCompression(onoff=1)
    canvas.setPageTransition(self, effectname=None, duration=1,
                               direction=0,dimension='H',motion='I')



