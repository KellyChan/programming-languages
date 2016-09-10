#
# Author: Kelly Chan
# Date  : Aug 10 2013
#
# Project: Rweibo
#
#--------------------------------------------------------------------------#
# Word Cloud
#---#

setwd("G:\\eclipseWorkspace\\R\\library\\packages\\Rweibo")

library(wordcloud)

d = read.csv("output\\wordCloud.csv")

png("output\\wordCloud.png")

dd = tail(d, 150)
op = par(bg = "lightyellow")
# grayLevels = gray((dd$freq)/(max(dd$freq) + 140))
# wordcloud(dd$word, dd$freq, colors = grayLevels)
rainbowLevels = rainbow((dd$freq)/(max(dd$freq) - 10))
wordcloud(dd$word, dd$freq, col = rainbow(length(d$freq)))
par(op)

dev.off()

