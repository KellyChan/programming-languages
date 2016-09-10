#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: F Distribution
#
#--------------------------------------------------------------------------#

setwd("G:\\eclipseWorkspace\\R\\distribution")


n=100
x <- seq(-2,2.5,length.out=1000)

png(filename="output\\fDistribution.png")
plot(x, df(x,1,1,0), col="gray", lwd=2, type='l',
		xlim=c(-2,2), ylim=c(-10,2),
		xaxs="i"    , yaxs="i",
		xlab=''     , ylab='density',
		main="The F Density Distribution")

lines(x,df(x,1,1,2), col="red"   , lwd=2)
lines(x,dt(x,2,2,2), col="black" , lwd=2)
lines(x,dt(x,2,4,2), col="orange", lwd=2)

legend("bottomright",legend=paste("dfl=", c(1,1,2,2),
				                  " df2=",c(1,1,2,4),
								  " ncp=",c(0,2,2,2)),
		             col=c("gray", "red","black","orange"),
		             lwd=2)


dev.off()
