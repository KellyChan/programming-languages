# data decomposition - Face Demo
 
 
# 1. download image
download.file("https://spark-public.s3.amazonaws.com/dataanalysis/face.rda",destfile="./data/face.rda")
load("./data/face.rda")
image(t(faceData)[,nrow(faceData):1])
 
# 2. SVD
svd1 <- svd(scale(faceData))
plot(svd1$d^2/sum(svd1$d^2),pch=19,xlab="Singluar vector",ylab="Variance explained")
 
 
# 3. SVD - approximations
 
svd1 <- svd(scale(faceData))
# %*% is matrix multiplication
 
# Here svd1$d[1] is a constant
approx1 <- svd1$u[,1] %*% t(svd1$v[,1]) * svd1$d[1]
 
# In these examples we need to make the diagonal matrix out of d
approx5 <- svd1$u[,1:5] %*% diag(svd1$d[1:5])%*% t(svd1$v[,1:5])
approx10 <- svd1$u[,1:10] %*% diag(svd1$d[1:10])%*% t(svd1$v[,1:10])
 
# 4. SVD - plot approximations
par(mfrow=c(1,4))
image(t(faceData)[,nrow(faceData):1])
image(t(approx10)[,nrow(approx10):1])
image(t(approx5)[,nrow(approx5):1])
image(t(approx1)[,nrow(approx1):1])