# Hierarchical clustering
 
# sample -> variables(x,y) -> plot 
set.seed(1234); par(mar=c(0,0,0,0))
x <- rnorm(20,mean=rep(1:3,each=4),sd=0.5)
y <- rnorm(20,mean=rep(c(1,2,1),each=4),sd=0.5)
plot(x,y,col="blue",pch=19,cex=2)
text(x+0.05,y+0.05,labels=as.character(1:20))
 
 
# variables(x,y) -> data.frame(x,y) -> dist(dataframe) -> hclust(dist(dataframe))) -> plot
dataFrame <- data.frame(x=x,y=y)
distxy <- dist(dataFrame)
hClustering <- hclust(distxy)
plot(hClustering)
 
 
# variables(x,y) -> data.frame(x,y) -> as.matrix(dataframe) -> heatmap(matrix[])
dataFrame <- data.frame(x=x,y=y)
set.seed(1344)
dataMatrix <- as.matrix(dataFrame)[sample(1:20),]
heatmap(dataMatrix)