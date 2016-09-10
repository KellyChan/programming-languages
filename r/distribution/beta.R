#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Beta Distribution
#
#--------------------------------------------------------------------------#

setwd("G:\\eclipseWorkspace\\R\\distribution")


n=10000
x <- seq(-5,5,length.out=n)

png(filename="output\\betaDistribution.png")
plot(x, dbeta(x,0.5,0.5), col="gray", lwd=2, type='l',
		                  xlim=c(0,1), ylim=c(0,10),
		                  xaxs="i"    , yaxs="i",
		                  xlab=''     , ylab='density',
		                  main="The Beta Density Distribution")

lines(x,dbeta(x,5,1), col="red"   , lwd=2)
lines(x,dbeta(x,1,3), col="black" , lwd=2)
lines(x,dbeta(x,2,2), col="orange", lwd=2)
lines(x,dbeta(x,2,5), col="green",  lwd=2)

legend("top",legend=paste("a=", c(0.5,5,1,2,2),
				          " b=",c(0.5,1,3,2,5)),
		     col=c("gray", "red","black","orange","green"),
		     lwd=2)


dev.off()
