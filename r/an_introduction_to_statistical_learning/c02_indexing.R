#----------------------------------------------------#
# Book: An Introduction to Statstical Learning
# Chap: Introduction - Indexing
#
# Author: Kelly Chan
# Date: Mar 28 2014
#----------------------------------------------------#

setwd('G:/rFiles/An_Introduction_to_Statistical_Learning')

#----------------------------------------------------#
# matrix

A=matrix (1:16 ,4 ,4)
A
A[2,3]
A[c(1,3) ,c(2,4) ] # row 1,3, col 2, 4
A[1:3 ,2:4]
A[1:2 ,]
A[ ,1:2]
A[1,]
A[-c(1,3) ,]


dim(A)
