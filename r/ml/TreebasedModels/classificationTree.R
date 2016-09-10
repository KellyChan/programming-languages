#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Classification Tree
#
#--------------------------------------------------------------------------#

setwd("G:\\eclipseWorkspace\\R\\DMalgorithms")


# Classification Tree with rpart
library(rpart)

# grow tree 
fit <- rpart(Kyphosis ~ Age + Number + Start, method="class", 
		                                      data=kyphosis)

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, main="Classification Tree for Kyphosis")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postscript plot of tree 
post(fit, file = "output\\classificationTree.ps", 
		  title = "Classification Tree for Kyphosis")  
  
# prune the tree 
pfit<- prune(fit, cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
  
# plot the pruned tree 
plot(pfit, uniform=TRUE, 
      	   main="Pruned Classification Tree for Kyphosis")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)

post(pfit, file = "output\\pClassificationTree.ps", 
		   title = "Pruned Classification Tree for Kyphosis")  