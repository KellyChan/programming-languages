#----------------------------------------------------#
# Book: An Introduction to Statstical Learning
# Chap: Introduction - Graphics
#
# Author: Kelly Chan
# Date: Mar 28 2014
#----------------------------------------------------#

setwd('G:/rFiles/An_Introduction_to_Statistical_Learning')

#----------------------------------------------------#
# plot

x=rnorm (100)
y=rnorm (100)
plot(x,y)
plot(x,y,
     xlab=" this is the x-axis",
     ylab=" this is the y-axis",
     main=" Plot of X vs Y")

#----------------------------------------------------#
# save images

pdf ("outputs/Figure.pdf")
plot(x,y,col =" green ")
dev.off ()

#----------------------------------------------------#
# contour

x=seq (1 ,10)
x=1:10
x=seq(-pi ,pi ,length =50)
x
y=x
f=outer(x,y,function(x,y) cos(y)/(1+x^2))
f

contour (x,y,f)
contour (x,y,f,nlevels =45, add=T)

#----------------------------------------------------#
# persp

fa=(f-t(f))/2
contour (x,y,fa,nlevels =15)


image(x,y,fa)
persp(x,y,fa)
persp(x,y,fa ,theta =30)
persp(x,y,fa ,theta =30, phi =20)
persp(x,y,fa ,theta =30, phi =70)
persp(x,y,fa ,theta =30, phi =40)
