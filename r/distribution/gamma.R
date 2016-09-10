#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Gamma Distribution
#
#--------------------------------------------------------------------------#


setwd("G:\\eclipseWorkspace\\R\\distribution")


n=100
x <- seq(0,10,length.out=n)

png(filename="output\\gammaDistribution.png")
plot(x, dgamma(x,1,2), col="gray", lwd=2, type='l',
		               xlim=c(0,10), ylim=c(0,2),
		               xaxs="i"    , yaxs="i",
		               xlab=''     , ylab='density',
		               main="The Gamma Density Distribution")

lines(x,dgamma(x, 2, 2), col="red"   , lwd=2)
lines(x,dgamma(x, 3, 2), col="black" , lwd=2)
lines(x,dgamma(x, 5, 1), col="orange", lwd=2)
lines(x,dgamma(x, 9, 1), col="green" , lwd=2)

legend("topright",legend=paste("shape=", c(1,2,3,5,9),
				               " rate=", c(2,2,2,1,1)
		                      ), 
		          col=c("gray", "red","black","orange","green"),
		          lwd=2)


dev.off()
