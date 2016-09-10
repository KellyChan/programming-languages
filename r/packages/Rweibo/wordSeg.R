#
# Author: Kelly Chan
# Date  : Aug 10 2013
#
# Project: Rweibo
#
#--------------------------------------------------------------------------#
# Load Data
#---#

setwd("G:\\eclipseWorkspace\\R\\library\\packages\\Rweibo")

library(Rweibo)

# Load data from Sina Weibo
#res <- web.search.content("eBay", page = 20, sleepmean = 10, sleepsd = 1)$Weibo
#write.csv(res, "output\\res.csv",row.names=TRUE)

#--------------------------------------------------------------------------#
# Parsing - word segmentation
#---#

res <- read.csv("output\\res.csv")

library(Rwordseg)

# Parsing
n = length(res[, 1])
res = res[res!=" "]
parseWords = unlist(lapply(X = res, FUN = segmentCN))

# Frequency Count with decreasing
freParseWords <- table(parseWords)
freParseWords <- sort(freParseWords, decreasing = T)

# Saved as data.frame
freParseWords <- data.frame(word = names(freParseWords), 
		freq = freParseWords)

write.csv(freParseWords, "output\\freParseWords.csv", row.names=TRUE)
