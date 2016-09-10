#----------------------------------------------------#
# Book: An Introduction to Statstical Learning
# Chap: Introduction - Indexing
#
# Author: Kelly Chan
# Date: Mar 28 2014
#----------------------------------------------------#

setwd('G:/rFiles/An_Introduction_to_Statistical_Learning')

#----------------------------------------------------#
# data

Auto=read.table ("Auto.data ")
fix(Auto)

Auto=read.table ("Auto.data", header =T,na.strings ="?")
fix(Auto)

Auto=read.csv (" Auto.csv", header =T,na.strings ="?")
fix(Auto)
dim(Auto)

Auto=na.omit(Auto)
dim(Auto)

names(Auto)