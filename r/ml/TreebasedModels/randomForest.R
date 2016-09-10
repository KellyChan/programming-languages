#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Frandom Forest
#
#--------------------------------------------------------------------------#

# Random Forest prediction of Kyphosis data
library(rpart)
library(randomForest)

fit <- randomForest(Kyphosis ~ Age + Number + Start,   data=kyphosis)
print(fit) # view results 
importance(fit) # importance of each predictor
