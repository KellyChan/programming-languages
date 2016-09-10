#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Normal Distribution
#
#--------------------------------------------------------------------------#


setwd("G:\\eclipseWorkspace\\R\\distribution")


n=100
x <- seq(-10,10,length.out=n)

png(filename="output\\normalDistribution.png")
plot(x, dnorm(x,0,1),col="red", lwd=2, type='l',
		             xlim=c(-5,5), ylim=c(0,1),
		             xaxs="i"    , yaxs="i",
					 xlab=''     , ylab='density',
		             main="The Normal Density Distribution")

lines(x,dnorm(x, 0,0.5),col="green" ,lwd=2)
lines(x,dnorm(x, 0,  2),col="gray"  ,lwd=2)
lines(x,dnorm(x,-2,  1),col="orange",lwd=2)

legend("topright",legend=paste("m="  , c(0,0,0,-2),
				               " sd=", c(1,0.5,2,1)
			                   ), 
				  col=c("red", "green","gray","orange"),
				  lwd=2)

		  
dev.off()

