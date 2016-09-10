#
# Author: Kelly Chan
# Date  : Aug 6 2013
#
# Project: Detecting Fraudulent Transactions
#
#--------------------------------------------------------------------------#
# Loading Data Into R
#----------------------#

library(DMwR)
data(sales)
head(sales)


#--------------------------------------------------------------------------#
# Exploring the Dataset
#----------------------#

summary(sales)

nlevels(sales$ID)
nlevels(sales$Prod)

length(which(is.na(sales$Quant) & is.na(sales$Val)))
sum(is.na(sales$Quant) & is.na(sales$Val))

table(sales$Insp)/nrow(sales) * 100


# Transactions per salespeople/product

totS <- table(sales$ID)
totP <- table(sales$Prod)
barplot(totS, main = "Transactions per salespeople", 
		      names.arg = "",
		      xlab = "Salespeople", 
			  ylab = "Amount")
barplot(totP, main = "Transactions per product", 
		      names.arg = "",
		      xlab = "Products", 
			  ylab = "Amount")
	  
	  
sales$Uprice <- sales$Val/sales$Quant
summary(sales$Uprice)

attach(sales)
upp <- aggregate(Uprice,list(Prod),median,na.rm=T)
topP <- sapply(c(T,F),function(o)upp[order(upp[,2],decreasing=o)[1:5],1])
colnames(topP) <- c('Expensive','Cheap')
topP

tops <- sales[Prod %in% topP[1, ], c("Prod", "Uprice")]
tops$Prod <- factor(tops$Prod)
boxplot(Uprice ~ Prod, data = tops, 
		               ylab = "Uprice", 
					   log = "y")
	
			   
vs <- aggregate(Val,list(ID),sum,na.rm=T)
scoresSs <- sapply(c(T,F),function(o) vs[order(vs$x,decreasing=o)[1:5],1])
colnames(scoresSs) <- c('Most','Least')
scoresSs			   


sum(vs[order(vs$x, decreasing = T)[1:100], 2])/sum(Val, na.rm = T) * 100
sum(vs[order(vs$x, decreasing = F)[1:2000], 2])/sum(Val,na.rm = T) * 100


qs <- aggregate(Quant,list(Prod),sum,na.rm=T)
scoresPs <- sapply(c(T,F),function(o) 
			                          qs[order(qs$x,decreasing=o)[1:5],1])
colnames(scoresPs) <- c('Most','Least')
scoresPs


sum(as.double(qs[order(qs$x,decreasing=T)[1:100],2]))/
sum(as.double(Quant),na.rm=T)*100


sum(as.double(qs[order(qs$x,decreasing=F)[1:4000],2]))/
sum(as.double(Quant),na.rm=T)*100

out <- tapply(Uprice,list(Prod=Prod),
		             function(x) length(boxplot.stats(x)$out))
out[order(out, decreasing = T)[1:10]]		
sum(out)

sum(out)/nrow(sales) * 100

#--------------------------------------------------------------------------#
# Data Problems
#----------------------#


# Missing Values

totS <- table(ID)
totP <- table(Prod)

nas <- sales[which(is.na(Quant) & is.na(Val)), c("ID", "Prod")]

propS <- 100 * table(nas$ID)/totS
propS[order(propS, decreasing = T)[1:10]]

propP <- 100 * table(nas$Prod)/totP
propP[order(propP, decreasing = T)[1:10]]


detach(sales)
sales <- sales[-which(is.na(sales$Quant) & is.na(sales$Val)),]

nnasQp <- tapply(sales$Quant,list(sales$Prod),
		                     function(x) sum(is.na(x)))
propNAsQp <- nnasQp/table(sales$Prod)
propNAsQp[order(propNAsQp,decreasing=T)[1:10]]

sales <- sales[!sales$Prod %in% c("p2442", "p2443"), ]
nlevels(sales$Prod)

sales$Prod <- factor(sales$Prod)
nlevels(sales$Prod)

nnasQs <- tapply(sales$Quant, list(sales$ID), function(x) sum(is.na(x)))
propNAsQs <- nnasQs/table(sales$ID)
propNAsQs[order(propNAsQs, decreasing = T)[1:10]]

nnasVp <- tapply(sales$Val,list(sales$Prod),
		                    function(x) sum(is.na(x)))
propNAsVp <- nnasVp/table(sales$Prod)
propNAsVp[order(propNAsVp,decreasing=T)[1:10]]

nnasVs <- tapply(sales$Val, list(sales$ID), function(x) sum(is.na(x)))
propNAsVs <- nnasVs/table(sales$ID)
propNAsVs[order(propNAsVs, decreasing = T)[1:10]]


tPrice <- tapply(sales[sales$Insp != "fraud", "Uprice"],
		               list(sales[sales$Insp != "fraud", "Prod"]), 
		               median, 
		               na.rm = T)
			   
noQuant <- which(is.na(sales$Quant))
sales[noQuant,'Quant'] <- ceiling(sales[noQuant,'Val'] /
						  tPrice[sales[noQuant,'Prod']])
noVal <- which(is.na(sales$Val))
sales[noVal,'Val'] <- sales[noVal,'Quant'] *
					  tPrice[sales[noVal,'Prod']]		
			  
sales$Uprice <- sales$Val/sales$Quant

save(sales, file = "salesClean.Rdata")


# Few Transactions of Some Products

attach(sales)
notF <- which(Insp != 'fraud')
ms <- tapply(Uprice[notF],list(Prod=Prod[notF]),function(x) {
			                                    bp <- boxplot.stats(x)$stats
			                                    c(median=bp[3],iqr=bp[4]-bp[2])
			                                    })
ms <- matrix(unlist(ms),
		     length(ms),
			 2,
			 byrow=T,
			 dimnames=list(names(ms),c('median','iqr')))
head(ms)


par(mfrow = c(1, 2))
plot(ms[, 1], ms[, 2], xlab = "Median", ylab = "IQR", main = "")
plot(ms[, 1], ms[, 2], xlab = "Median", ylab = "IQR", main = "",
		                                col = "grey", log = "xy")
smalls <- which(table(Prod) < 20)
points(log(ms[smalls, 1]), log(ms[smalls, 2]), pch = "+")


dms <- scale(ms)
smalls <- which(table(Prod) < 20)
prods <- tapply(sales$Uprice, sales$Prod, list)
similar <- matrix(NA, length(smalls), 7, 
		              dimnames = list(names(smalls),
				      c("Simil", "ks.stat", "ks.p", "medP", "iqrP", "medS","iqrS")))

for (i in seq(along = smalls)) 
{
	  d <- scale(dms, dms[smalls[i], ], FALSE)
	  d <- sqrt(drop(d^2 %*% rep(1, ncol(d))))
	  stat <- ks.test(prods[[smalls[i]]], prods[[order(d)[2]]])
	  similar[i, ] <- c(order(d)[2], stat$statistic, 
			                         stat$p.value,
			                         ms[smalls[i], ], 
									 ms[order(d)[2], ])
}

head(similar)

levels(Prod)[similar[1, 1]]

nrow(similar[similar[, "ks.p"] >= 0.9, ])
sum(similar[, "ks.p"] >= 0.9)
save(similar, file = "similarProducts.Rdata")


#--------------------------------------------------------------------------#
# Defining the Data Mining Tasks
#----------------------#

library(ROCR)
data(ROCR.simple)
pred <- prediction(ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred, "prec", "rec")
plot(perf)

PRcurve <- function(preds, trues, ...) 
{
	        require(ROCR, quietly = T)
	        pd <- prediction(preds, trues)
	        pf <- performance(pd, "prec", "rec")
	        pf@y.values <- lapply(pf@y.values, function(x) rev(cummax(rev(x))))
	        plot(pf, ...)
}

PRcurve(ROCR.simple$predictions, ROCR.simple$labels)

pred <- prediction(ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred, "lift", "rpp")
plot(perf, main = "Lift Chart")


CRchart <- function(preds, trues, ...) 
{
	         require(ROCR, quietly = T)
	         pd <- prediction(preds, trues)
	         pf <- performance(pd, "rec", "rpp")
	         plot(pf, ...)
}

CRchart(ROCR.simple$predictions, 
		ROCR.simple$labels,
		main='Cumulative Recall Chart')

avgNDTP <- function(toInsp,train,stats) 
{
	       if (missing(train) && missing(stats))
			   stop('Provide either the training data or the product stats')
	       
		   if (missing(stats))
		   {
			   notF <- which(train$Insp != 'fraud')
		       stats <- tapply(train$Uprice[notF],list(Prod=train$Prod[notF]),
				                function(x) 
								{
									bp <- boxplot.stats(x)$stats
					                c(median=bp[3],iqr=bp[4]-bp[2])
					           })
		       stats <- matrix(unlist(stats),
				               length(stats),
						       2,
						       byrow=T,
				               dimnames=list(names(stats),c('median','iqr'))
				              )
		       stats[which(stats[,'iqr']==0),'iqr'] <-
		       stats[which(stats[,'iqr']==0),'median']
           }

           mdtp <- mean(abs(toInsp$Uprice-stats[toInsp$Prod,'median']) /
				   stats[toInsp$Prod,'iqr'])
           return(mdtp)
}


evalOutlierRanking <- function(testSet,rankOrder,Threshold,statsProds) 
{
	  ordTS <- testSet[rankOrder,]
	  N <- nrow(testSet)
	  nF <- if (Threshold < 1) as.integer(Threshold*N) else Threshold
      cm <- table(c(rep('fraud',nF),rep('ok',N-nF)),ordTS$Insp)
	  prec <- cm['fraud','fraud']/sum(cm['fraud',])
	  rec <- cm['fraud','fraud']/sum(cm[,'fraud'])
	  AVGndtp <- avgNDTP(ordTS[nF,],stats=statsProds)
	  return(c(Precision=prec,Recall=rec,avgNDTP=AVGndtp))
}


#--------------------------------------------------------------------------#
# Obtaining Outlier Rankings
#----------------------#


BPrule <- function(train,test) 
{
	  notF <- which(train$Insp != 'fraud')
	  ms <- tapply(train$Uprice[notF],list(Prod=train$Prod[notF]),
			                          function(x) 
									  {
				                        bp <- boxplot.stats(x)$stats
				                        c(median=bp[3],iqr=bp[4]-bp[2])
				                      }
				  )
	  ms <- matrix(unlist(ms),length(ms),
			                  2,
							  byrow=T,
			                  dimnames=list(names(ms),c('median','iqr')))
	  ms[which(ms[,'iqr']==0),'iqr'] <- ms[which(ms[,'iqr']==0),'median']
	  ORscore <- abs(test$Uprice-ms[test$Prod,'median']) /
			     ms[test$Prod,'iqr']
	  
	  return(list(rankOrder=order(ORscore,decreasing=T),
			 rankScore=ORscore))
}

notF <- which(sales$Insp != 'fraud')
globalStats <- tapply(sales$Uprice[notF],list(Prod=sales$Prod[notF]),
		              function(x) 
					  {
						  bp <- boxplot.stats(x)$stats
			              c(median=bp[3],iqr=bp[4]-bp[2])
			          })
globalStats <- matrix(unlist(globalStats),
		              length(globalStats),
					  2,
					  byrow=T,
		              dimnames=list(names(globalStats),c('median','iqr')))
globalStats[which(globalStats[,'iqr']==0),'iqr'] <-
globalStats[which(globalStats[,'iqr']==0),'median']

ho.BPrule <- function(form, train, test, ...) 
{
	  res <- BPrule(train,test)
	  structure(evalOutlierRanking(test,res$rankOrder,...),
			    itInfo=list(preds=res$rankScore,
						    trues=ifelse(test$Insp=='fraud',1,0)
				           )
			    )
}

bp.res <- holdOut(learner('ho.BPrule',
				          pars=list(Threshold=0.1,
				          statsProds=globalStats)),
		          dataset(Insp ~ .,sales),
		          hldSettings(3,0.3,1234,T),
		          itsInfo=TRUE
				 )
				 
summary(bp.res)

par(mfrow=c(1,2))
info <- attr(bp.res,'itsInfo')
PTs.bp <- aperm(array(unlist(info),
				dim=c(length(info[[1]]),2,3)),
		        c(1,3,2)
			   )
PRcurve(PTs.bp[,,1],PTs.bp[,,2],
		main='PR curve',
		avg='vertical')
CRchart(PTs.bp[,,1],PTs.bp[,,2],
		main='Cumulative Recall curve',
		avg='vertical')

