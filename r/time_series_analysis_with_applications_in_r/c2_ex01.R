#----------------------------------------------------#
# Book: Time Series Analysis with Application in R
# Chap: Trends
#
# Author: Kelly Chan
# Date: Mar 27 2014
#----------------------------------------------------#


setwd('G:\rFiles\Time_Series_Analysis_with_Applications_in_R')

library(TSA)

#----------------------------------------------------#
# Random Walk: Linear Model
#-------#

data(rwalk)

model1 = lm(rwalk ~ time(rwalk))
summary(model1)

plot(rwalk, type='o', ylab='Random Walk')
abline(model1)

plot(y=rstudnet(model1), x=as.vector(time(rwalk)),
     ylab='Standardized Residuals', xlab='Time', type='o')

plot(y=rstudent(model1), x=fitted(model1),
     ylab='Standaridized Residuals', xlab='Fitted Trend Line Values',
     type='p')

acf(rstudent(model1))

#----------------------------------------------------#
# Temperature: Seasonal
#-------#

data(tempdub)
month. = season(tempdub) # set period
model2 = lm(tempdub ~ month.-1) # -1 removes the intercept term
summary(model2)

model3 = lm(tempdub ~ month.) # January is dropped automatically
summary(model3)

plot(y=rstudent(model3), x=as.vector(time(tempdub)),
     ylab='Standardized Residuals', xlab='Time', 
     type='o')

plot(y=rstudent(model3), x=as.vector(time(tempdub)),
     ylab='Standardized Residuals', xlab='Time', 
     type='l')
#points(y=rstudent(model3), x=as.vector(time(tempdub)),
#       pch=as.vector(season(tempdub)))

hist(rstudent(model3), xlab='Standarized Residuals')
qqnorm(rstudent(model3))

acf(rstudent(model3))

#----------------------------------------------------#
# Temperature: Consine
#-------#

har. = harmonic(tempdub, 1)
model4 = lm(tempdub ~ har.)
summary(model4)

plot(ts(fitted(model4), freq=12, start=c(1946,1)), ylab='Temperature', type='l')
ylim = range(c(fitted(model4), tempdub)); points(tempdub)

