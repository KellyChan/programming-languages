#----------------------------------------------------#
# Book: An Introduction to Statstical Learning
# Chap: Introduction - Basic Commands
#
# Author: Kelly Chan
# Date: Mar 28 2014
#----------------------------------------------------#


#----------------------------------------------------#
# values

x <- c(1,3,2,5)
x

x = c(1,6,2)
x

y = c(1,4,3)

#----------------------------------------------------#
# length

length (x)
length (y)

#----------------------------------------------------#
# list variables

ls()
rm(x,y)  # delete
ls()
rm(list=ls())
ls()

#----------------------------------------------------#
# Matrix

?matrix
x=matrix (data=c(1,2,3,4) , nrow=2, ncol =2)
x

x=matrix (c(1,2,3,4) ,2,2)
x

matrix (c(1,2,3,4) ,2,2,byrow =TRUE)

#----------------------------------------------------#
# Math

sqrt(x)
x^2

#----------------------------------------------------#
# Distribution

x=rnorm (50)
x
y=x+rnorm (50, mean=50, sd=.1)
y

cor(x,y)

#----------------------------------------------------#
# Stat

set.seed (1303)
rnorm(50)

set.seed (3)
y=rnorm (100)
mean(y)
var(y)
sqrt(var(y))
sd(y)
