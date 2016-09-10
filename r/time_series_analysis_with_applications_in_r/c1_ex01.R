#----------------------------------------------------#
# Book: Time Series Analysis with Application in R
# Chap: Time Series
#
# Author: Kelly Chan
# Date: Mar 27 2014
#----------------------------------------------------#

setwd('G:\rFiles\Time_Series_Analysis_with_Applications_in_R')

library(TSA)

#----------------------------------------------------#
# Rain 
#-------#

win.graph(width=4.875, height=2.5, pointsize=8)
data(larain)
larain
plot(larain, ylab='Inches', xlab='Year', type='o')


win.graph(width=3, height=3, pointsize=8)
plot(y=larain, x=zlag(larain), 
     ylab='Inches', xlab='Previous Year Inches')

#----------------------------------------------------#
# Industry
#-------#

win.graph(width=4.875, height=2.5, pointsize=8)
data(color)
plot(color, ylab='Color Property', xlab='Batch', type='o')

win.graph(width=3, height=3, pointsize=8)
plot(y=color, x=zlag(color),
     ylab='Color Property', xlab='Previous Batch Color Property')

#----------------------------------------------------#
# Hare
#-------#

data(hare)
plot(hare, ylab='Abundance', xlab='Year', type='o')

plot(y=hare, x=zlag(hare),
     ylab='Abundance', xlab='Previous Year Abundance')


#----------------------------------------------------#
# Temperature
#-------#

data(tempdub)
plot(tempdub, ylab='Temperature', type='o')

#----------------------------------------------------#
# Oil
#-------#

data(oilfilters)
plot(oilfilters, ylab='Sales', type='o')
plot(oilfilters, ylab='Sales', type='l')

