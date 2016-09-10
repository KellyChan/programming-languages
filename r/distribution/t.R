#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: F Distribution
#
#--------------------------------------------------------------------------#

setwd("G:\\eclipseWorkspace\\R\\distribution")


n=100
x <- seq(-5,5,length.out=n)

png(filename="output\\tDistribution.png")
plot(x, dt(x,1,0), col="gray", lwd=2, type='l',
		           xlim=c(-5,5), ylim=c(0,0.5),
		           xaxs="i"    , yaxs="i",
		           xlab=''     , ylab='density',
		           main="The T Density Distribution")

lines(x,dt(x,5,0),  col="red"   , lwd=2)
lines(x,dt(x,5,2),  col="black" , lwd=2)
lines(x,dt(x,50,2), col="orange", lwd=2)

legend("topleft",legend=paste("dfl=", c(1,5,5,50),
				              " ncp=",c(0,0,2,2)),
		                col=c("gray", "red","black","orange"),
		                lwd=2)
				

dev.off()
