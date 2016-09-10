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
#res <- web.search.content("eBay", page = 5, sleepmean = 10, sleepsd = 1)$Weibo
#write.csv(res, "output\\res.csv",row.names=TRUE)

#--------------------------------------------------------------------------#
# Corpus created (data cleaning) and Word parsing
#---#

res <- read.csv("output\\res.csv")

# 1. Corpus created and data cleaning

library(tm)

# create Corpus
weiboCorpus <- Corpus(VectorSource(res))

# data cleaning
# clean punctuation, numbers, stop words, lowercase -- depends on data story
#weiboCorpus <- tm_map(weiboCorpus,removePunctuation)
#weiboCorpus <- tm_map(weiboCorpus,removeNumbers)
#weiboCorpus <- tm_map(weiboCorpus, removeWords, stopwords("english"))
weiboCorpus <- tm_map(weiboCorpus, tolower)

# clean URL
removeURL <- function(x) gsub("http[[:alnum:]]*","",x)
weiboCorpus <- tm_map(weiboCorpus,removeURL)

# clean stop words
removeStopWords <- function(x) gsub("µÄ","",x)
weiboCorpus <- tm_map(weiboCorpus,removeStopWords)

removeStopWords <- function(x) gsub("ÔÚ","",x)
weiboCorpus <- tm_map(weiboCorpus,removeStopWords)



# 2. word parsing

library(Rwordseg)

# insert extra words not in the library
insertWords(c("ÌÔ±¦","ÌÔ±¦µê","¹È¸èÑÛ¾µ","Google Glass","¹È¸è"
			  "ËÕÄþÒ×¹º","ÌìÃ¨","ÈËÁ³Ê¶±ð"))

# Parsing
weiboCorpus = weiboCorpus[weiboCorpus!=" "]
parseWords = unlist(lapply(X = weiboCorpus, FUN = segmentCN))

# Frequency Count with decreasing
freParseWords <- table(parseWords)
freParseWords <- sort(freParseWords, decreasing = T)


# Saved as data.frame
freParseWords <- data.frame(word = names(freParseWords), 
		                    freq = freParseWords)

write.csv(freParseWords, "output\\freParseWordsCleaned.csv", row.names=TRUE)
