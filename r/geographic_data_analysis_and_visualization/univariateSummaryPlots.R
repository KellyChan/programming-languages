#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Geographic Data Analysis and Visualization
# Source: http://geography.uoregon.edu/GeogR/topics/index.html
#
# example: Univariate Summary Plots
# source: http://geography.uoregon.edu/GeogR/topics/univariateplots02.htm
#
#--------------------------------------------------------------------------#
# Set Work Directory / Load data

setwd("G:\\eclipseWorkspace\\R\\examples\\geographicDataAnalysisAndVisualization")

specmap <- read.csv("data\\specmap.csv")
scanvote <- read.csv("data\\scanvote.csv")

#--------------------------------------------------------------------------#
# Plotting the data
#----#

attach(specmap)
# Histograms
png("output\\hist.png")
hist(O18)
dev.off()

png("output\\histBreak.png")
hist(O18, breaks=20)
dev.off()

# Density Plots (or Kernel Plots/Smoothed Histograms)
png("output\\densityPlot.png")
O18.density <- density(O18)
plot(O18.density)
dev.off()

# Histograms + Density Plots
png("output\\histDensity.png")
O18.density <- density(O18)
hist(O18, breaks=40, probability=TRUE)
lines(O18.density)
rug(O18)
dev.off()

detach(specmap)

#--------------------------------------------------------------------------#

# Boxplot (or Box-and-Whisker Plot)
attach(scanvote)

png("output\\boxplot.png")
boxplot(Pop)
dev.off()

png("output\\bloxplotLog.png")
boxplot(log10(Pop))
dev.off()

# QQ Plot (or QQ Normal Plot)
png("output\\qqPlot.png")
qqnorm(Pop)
qqline(Pop)
dev.off()

png("output\\qqPlotLog.png")
qqnorm(log10(Pop))
qqline(log10(Pop))
dev.off()

detach(scanvote)

#--------------------------------------------------------------------------#
# An Aside on Reference Distributions

z <- seq(-3.0,3.0,.05)
pdf.z <- dnorm(z)   # get probability density function
png("output\\probabilityDensity.png")
plot(z, pdf.z)
dev.off()

png("output\\cumulativeDistribution.png")
cdf.z <- pnorm(z)   # get cumulative distribution function
plot(z, cdf.z)
dev.off()

png("output\\invCdf.png")
p <- seq(0,1,.01)
invcdf.z <- qnorm(p)  # get inverse cumulative distibution function
plot(p,invcdf.z)
dev.off()
