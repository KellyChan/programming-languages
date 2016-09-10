#----------------------------------------------------#
# Book: An Introduction to Statstical Learning
# Chap: Introduction - Modeling
#
# Author: Kelly Chan
# Date: Mar 28 2014
#----------------------------------------------------#

setwd('G:/rFiles/An_Introduction_to_Statistical_Learning')

#----------------------------------------------------#
# plot

plot(cylinders , mpg)

plot(Auto$cylinders , Auto$mpg )

attach (Auto)
plot(cylinders , mpg)

cylinders =as.factor (cylinders )
plot(cylinders , mpg)
plot(cylinders , mpg , col ="red ")
plot(cylinders , mpg , col ="red", varwidth =T)
plot(cylinders , mpg , col ="red", varwidth =T,horizontal =T)
plot(cylinders , mpg , col ="red", varwidth =T, xlab=" cylinders ", ylab ="MPG ")

#----------------------------------------------------#
# hist

hist(mpg)
hist(mpg ,col =2)
hist(mpg ,col =2, breaks =15)

#----------------------------------------------------#
# combination plot

pairs(Auto)
pairs(∼ mpg + displacement + horsepower + weight + acceleration , Auto)
plot(horsepower ,mpg)
identify (horsepower ,mpg ,name)

#----------------------------------------------------#
# summary

summary (Auto)
summary (mpg)