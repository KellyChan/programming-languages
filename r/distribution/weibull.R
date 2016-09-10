#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Weibull Distribution
#
#--------------------------------------------------------------------------#


setwd("G:\\eclipseWorkspace\\R\\distribution")


n=100
x <- seq(0,2.5,length.out=1000)

png(filename="output\\weibullDistribution.png")
plot(x, dweibull(x,.5), col="gray", lwd=2, type='l',
		                xlim=c(0,2.5), ylim=c(0,2.5),
		                xaxs="i"    , yaxs="i",
		                xlab=''     , ylab='density',
		                main="The Weibull Density Distribution")

lines(x,dweibull(x, 1), col="red"   , lwd=2)
lines(x,dweibull(x,1.5), col="black" , lwd=2)
lines(x,dweibull(x, 5), col="orange", lwd=2)

legend("topright",legend=paste("shape=", c(.5,1,1.5,5)), 
		          col=c("gray", "red","black","orange"),
		          lwd=2)


dev.off()
