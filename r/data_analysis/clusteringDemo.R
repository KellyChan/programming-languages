download.file("https://dl.dropbox.com/u/7710864/courseraPublic/samsungData.rda"
              ,destfile="./data/samsungData.rda",method="curl")
load("./data/samsungData.rda")
names(samsungData)[1:12]
 
table(samsungData$activity)
 
 
 
# 1.1 Average Acceleration: plotting
 
par(mfrow=c(1,2))
numericActivity <- as.numeric(as.factor(samsungData$activity))[samsungData$subject==1]
 
plot(samsungData[samsungData$subject==1,1],
     pch=19,
     col=numericActivity,
     ylab=names(samsungData)[1])
 
plot(samsungData[samsungData$subject==1,2],
     pch=19,
     col=numericActivity,
     ylab=names(samsungData)[2])
 
legend(150,-0.1,
       legend=unique(samsungData$activity),
       col=unique(numericActivity),
       pch=19)
 
 
# 1.2. Average Acceleration: hierarchical clustering
 
source("http://dl.dropbox.com/u/7710864/courseraPublic/myplclust.R")
distanceMatrix <- dist(samsungData[samsungData$subject==1,1:3])
hclustering <- hclust(distanceMatrix)
myplclust(hclustering,lab.col=numericActivity)
 
 
 
# 2.1 Max Acceleration: plotting
par(mfrow=c(1,2))
plot(samsungData[samsungData$subject==1,10],
     pch=19,
     col=numericActivity,
     ylab=names(samsungData)[10])
 
plot(samsungData[samsungData$subject==1,11],
     pch=19,
     col=numericActivity,
     ylab=names(samsungData)[11])
 
 
# 2.2 Max Accerleration: hierarchical cluserting
 
source("http://dl.dropbox.com/u/7710864/courseraPublic/myplclust.R")
distanceMatrix <- dist(samsungData[samsungData$subject==1,10:12])
hclustering <- hclust(distanceMatrix)
myplclust(hclustering,lab.col=numericActivity)
 
 
# 3.1 Data Decomposition: SVD and plotting
 
svd1 = svd(scale(samsungData[samsungData$subject==1,-c(562,563)]))
par(mfrow=c(1,2))
plot(svd1$u[,1],col=numericActivity,pch=19)
plot(svd1$u[,2],col=numericActivity,pch=19)
 
 
# 3.2 Data Decomposition: maximum contributor
plot(svd1$v[,2],pch=19)
 
 
# 3.3 Maximum Contributor: new hierarchical clustering
 
maxContrib <- which.max(svd1$v[,2])
distanceMatrix <- dist(samsungData[samsungData$subject==1,c(10:12,maxContrib)])
hclustering <- hclust(distanceMatrix)
myplclust(hclustering,lab.col=numericActivity)
 
names(samsungData)[maxContrib]
 
 
 
# 4.11 K-Means Clustering: nstart=1, first try
 
kClust <- kmeans(samsungData[samsungData$subject==1,-c(562,563)],centers=6)
table(kClust$cluster,samsungData$activity[samsungData$subject==1])
 
 
# 4.12 K-Means Clustering: nstart=1, second try
 
kClust <- kmeans(samsungData[samsungData$subject==1,-c(562,563)],centers=6,nstart=1)
table(kClust$cluster,samsungData$activity[samsungData$subject==1])
 
 
# 4.21 K-Means Clustering: nstart=100, first try
 
kClust <- kmeans(samsungData[samsungData$subject==1,-c(562,563)],centers=6,nstart=100)
table(kClust$cluster,samsungData$activity[samsungData$subject==1])
 
# 4.22 K-Means Clustering: nstart=100, second try
 
kClust <- kmeans(samsungData[samsungData$subject==1,-c(562,563)],centers=6,nstart=100)
table(kClust$cluster,samsungData$activity[samsungData$subject==1])
 
 
# 4.31 Cluster 1 Variable Centers (laying)
plot(kClust$center[1,1:10],pch=19,ylab="Cluster Center",xlab="")
 
# 4.32 Cluster 1 Variable Centers (walking)
plot(kClust$center[6,1:10],pch=19,ylab="Cluster Center",xlab="")