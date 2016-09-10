#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Regression Tree
#
#--------------------------------------------------------------------------#

setwd("G:\\eclipseWorkspace\\R\\DMalgorithms")

# Regression Tree Example
library(rpart)

# grow tree 
fit <- rpart(Mileage~Price + Country + Reliability + Type, method="anova", 
		                                                   data=cu.summary)

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# create additional plots 
par(mfrow=c(1,2)) # two plots on one page 
rsq.rpart(fit) # visualize cross-validation results  	

# plot tree 
plot(fit, uniform=TRUE, 
     	  main="Regression Tree for Mileage ")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postcript plot of tree 
post(fit, file = "output\\regressionTree.ps", 
		  title = "Regression Tree for Mileage ")
  
# prune the tree 
pfit<- prune(fit, cp=0.01160389) # from cptable   
  
# plot the pruned tree 
plot(pfit, uniform=TRUE, 
		   main="Pruned Regression Tree for Mileage")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)

post(pfit, file = "output\\pRegreesionTree.ps", 
		   title = "Pruned Regression Tree for Mileage")  