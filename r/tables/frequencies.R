#
# Author: Kelly Chan
# Date  : Aug 23 2013
#
# Project: Nokia - Mobile Phones
# Summary: holecount tables
# 
#---------------------------------------------------------------------!
# Database 
#----------!

library(RSQLite)
driver     <- dbDriver("SQLite")
connection <- dbConnect(driver, dbname="D:\\database\\SQLite\\business\\nokia.sqlite")

# send a query
mobilephones <- dbGetQuery(connection, "SELECT * from mobilephones")

dbDisconnect(connection)
dbUnloadDriver(driver)


setwd("G:\\eclipseWorkspace\\R\\business\\nokia")
attach(mobilephones)
#---------------------------------------------------------------------!
# Holecount Tables
#----------!

freqTable <- function(var,varName)
{
	var <- table(var)
	var <- sort(var, decreasing = T)
	write.csv(var, paste("outputs\\hc\\freq-",varName,".csv"),row.names=TRUE)
}

freqTable(Type,'Type')
freqTable(PhoneModel,'PhoneModel')
freqTable(ScreenType,'ScreenType')
freqTable(Released,'Released')	
freqTable(Quarter,'Quarter')	
freqTable(S,'S')	
freqTable(Technology,'Technology')	
freqTable(Platform,'Platform')	
freqTable(Generation,'Generation')	
freqTable(FormFactor,'FormFactor')	
freqTable(Ringtone,'Ringtone')	
freqTable(Camera,'Camera')	
freqTable(Notes,'Notes')

#--------------------------------------------------------------------------#
# Frequency Plot
#---#

freqPlot <- function(var,varName)
{
	var <- table(var)
	png(paste("outputs\\freqPlot\\",varName,".png"))
	plot(var, main=paste("Frequency - ",varName))
	dev.off()
}

freqPlot(Type,'Type')
freqPlot(PhoneModel,'PhoneModel')
freqPlot(ScreenType,'ScreenType')
freqPlot(Released,'Released')	
freqPlot(Quarter,'Quarter')	
freqPlot(S,'S')	
freqPlot(Technology,'Technology')	
freqPlot(Platform,'Platform')	
freqPlot(Generation,'Generation')	
freqPlot(FormFactor,'FormFactor')	
freqPlot(Ringtone,'Ringtone')	
freqPlot(Camera,'Camera')	
freqPlot(Notes,'Notes')

#---------------------------------------------------------------------!
detach(mobilephones)


