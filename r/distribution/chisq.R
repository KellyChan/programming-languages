#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Chisq Distribution
#
#--------------------------------------------------------------------------#

setwd("G:\\eclipseWorkspace\\R\\distribution")


n=1000
x <- seq(-5,5,length.out=n)

png(filename="output\\chisqDistribution.png")
plot(x, dchisq(x,1), col="gray", lwd=2, type='l',
		             xlim=c(0,5), ylim=c(0,2),
		             xaxs="i"    , yaxs="i",
		             xlab=''     , ylab='density',
	              	 main="The Chrisq Density Distribution")

lines(x,dchisq(x,2), col="red"   , lwd=2)
lines(x,dchisq(x,3), col="black" , lwd=2)
lines(x,dchisq(x,5), col="orange", lwd=2)

legend("topright",legend=paste("df=", c(1,2,3,5)),
		          col=c("gray", "red","black","orange"),
		          lwd=2)


dev.off()
