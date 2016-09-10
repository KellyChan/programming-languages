#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Conditional Inference Tree
#
#--------------------------------------------------------------------------#

# Conditional Inference Tree for Mileage
library(rpart) # get the data kyphosis & mileage
library(party) # package for ctree

fitKyphosis <- ctree(Kyphosis ~ Age + Number + Start, data=kyphosis)
png("output\\conditionalInferenceTreeKyphosis.png")
plot(fitKyphosis, main="Conditional Inference Tree for Kyphosis")
dev.off()

fitMileage <- ctree(Mileage~Price + Country + Reliability + Type, 
		            data=na.omit(cu.summary))
			
png("output\\conditionalInferenceTreeMileage.png")
plot(fitMileage, main="Conditional Inference Tree for Mileage")	
dev.off()

		  