#
# Author: Kelly Chan
# Date  : Aug 7 2013
#
# Project: Geographic Data Analysis and Visualization
# Source: http://geography.uoregon.edu/GeogR/topics/index.html
#
# example: Univariate Enumerative Plots
# source: http://geography.uoregon.edu/GeogR/topics/univariateplots01.htm
#
#--------------------------------------------------------------------------#
# Set Work Directory / Load data

setwd("G:\\eclipseWorkspace\\R\\examples\\geographicDataAnalysisAndVisualization")

cities <- read.csv("data\\cities.csv")
specmap <- read.csv("data\\specmap.csv")

#--------------------------------------------------------------------------#
# Plotting the data
#----#

# use large cities data, Index Plot 
attach(cities)
plot(Pop.2000)

# Y Zero High-Density Plot
png("output\\plotType.png")
plot(Pop.2000, type="h") #type = "l", "b", "o", "s", or "S"
dev.off()

# Time series plot
attach(specmap)
png("output\\plot.png")
plot(O18)
dev.off()

png("output\\plotYaxis.png")
plot(O18, ylim=c(2.5,-2.5))   #  invert y-axis
dev.off()

# Strip Plot (¡°univariate scatter diagram"
png("output\\stripchart.png")
stripchart(O18)
dev.off()

png("output\\stripchartStack.png")
stripchart(O18, method="stack")   # stack points to reduce overlap
dev.off()

# Dot Plot  / Dot Chart
png("output\\dotchart.png")
dotchart(Pop.2000, labels=City)
dev.off()

png("output\\dotchartwithIndex.png")
index <- order(Pop.2000)      #to use index to record the rank-order 
dotchart(Pop.2000[index], labels=City[index])
dev.off()

#--------------------------------------------------------------------------#
detach(specmap)
detach(cities)