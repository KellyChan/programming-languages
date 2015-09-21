setwd('path')

library(ggplot2)
library(scales)

#-----------------------------------------------------------------------------#
# Daily Visits

data <- read.csv('outputs/stat/tables/holecount-Date.csv', header=TRUE)
data$value <- as.Date(data$value)
p <- ggplot(data=data, aes(x=value, y=freq, group=1)) + 
  geom_line(colour="grey", size=1.5) + 
  geom_point(colour="grey", size=4, shape=21, fill="orange") +
  scale_x_date(breaks = "1 week", labels=date_format("%m-%d")) +
  ggtitle("Daily Visits") +
  xlab("Date") + 
  ylab("Frequency")
print(p)
ggsave(p, file='outputs/stat/pics/daily_visits.png')

#-----------------------------------------------------------------------------#
# Visits of weekday

data <- read.csv('outputs/stat/tables/holecount-Weekday.csv', header=TRUE)
data$value <- factor(data$value, levels=c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"))
data[order(data$value),]
print(data)
p <- ggplot(data=data, aes(x=value, y=freq, group=1)) +
  geom_bar(colour="grey", fill="grey", width=.7, stat="identity") +
  geom_line(colour="orange", size=1.2) +
  ggtitle("Visits of weekday") +
  xlab("Weekday") +
  ylab("Frequency")
print(p)
ggsave(p, file='outputs/stat/pics/visits_of_weekday.png')

#-----------------------------------------------------------------------------#
# Hourly visits

data <- read.csv('outputs/stat/tables/holecount-Hour.csv', header=TRUE)
print(data)
p <- ggplot(data=data, aes(x=value, y=freq, group=1)) + 
  geom_line(colour="grey", size=1.5) + 
  geom_point(colour="grey", size=4, shape=21, fill="orange") +
  scale_x_continuous(limits = c(0, 23), breaks = seq(0,23,1)) +
  ggtitle("Hourly Visits") +
  xlab("Time") + 
  ylab("Frequency")
print(p)
ggsave(p, file='outputs/stat/pics/hourly_visits.png')

