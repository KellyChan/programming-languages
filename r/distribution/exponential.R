#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Exponential Distribution
#
#--------------------------------------------------------------------------#


setwd("G:\\eclipseWorkspace\\R\\distribution")


n=100
x <- seq(-5,5,length.out=n)

png(filename="output\\exponentialDistribution.png")
plot(x, dexp(x,0.5),col="red", lwd=2, type='l',
		            xlim=c(0,2), ylim=c(0,5),
		            xaxs="i"    , yaxs="i",
		            xlab=''     , ylab='density',
		            main="The Exponential Density Distribution")

lines(x,dexp(x,1),col="orange" ,lwd=2)
lines(x,dexp(x,2),col="black"  ,lwd=2)
lines(x,dexp(x,5),col="gray"   ,lwd=2)

legend("topright",legend=paste("rate=", c(5,2,1,0.5)), 
		          col=c("gray", "black","orange","red"),
		          lwd=2)

dev.off()
