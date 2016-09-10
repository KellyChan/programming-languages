#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Logistic Regression
#
#--------------------------------------------------------------------------#

setwd("G:\\eclipseWorkspace\\R\\DMalgorithms")

# Logistic Regression
# where F is a binary factor and 
# x1-x3 are continuous predictors 
fit <- glm(F~x1+x2+x3,data=mydata,family=binomial())

summary(fit) # display results
confint(fit) # 95% CI for the coefficients
exp(coef(fit)) # exponentiated coefficients
exp(confint(fit)) # 95% CI for exponentiated coefficients

predict(fit, type="response") # predicted values
residuals(fit, type="deviance") # residuals
