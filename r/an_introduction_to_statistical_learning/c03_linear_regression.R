#----------------------------------------------------#
# Book: An Introduction to Statstical Learning
# Chap: Linear Regression
#
# Author: Kelly Chan
# Date: Mar 28 2014
#----------------------------------------------------#

setwd('G:/rFiles/An_Introduction_to_Statistical_Learning')

#----------------------------------------------------#
# library

library (MASS)  # data + function
library (ISLR)  # data

#----------------------------------------------------#
# simple linear regression

data(Boston)
fix(Boston)
names(Boston)

lm.fit =lm(medv ~ lstat, data=Boston)
lm.fit
summary (lm.fit)
names(lm.fit )
coef(lm.fit)
confint (lm.fit)

predict (lm.fit ,
         data.frame(lstat=(c(5 ,10 ,15) )),
         interval ="confidence")

predict (lm.fit,
         data.frame(lstat =(c(5 ,10 ,15) )),
         interval ="prediction")

plot(lstat ,medv)
abline (lm.fit)
abline (lm.fit ,lwd =3)
abline (lm.fit ,lwd =3, col ="red ")

plot(lstat ,medv ,col ="red ")

plot(lstat ,medv ,pch =20)
plot(lstat ,medv ,pch ="+")
plot (1:20 ,1:20, pch =1:20)

par(mfrow =c(2,2)) # divides pics with 2 * 2 grid
plot(lm.fit)
plot(predict (lm.fit), residuals (lm.fit))
plot(predict (lm.fit), rstudent (lm.fit))
plot(hatvalues (lm.fit ))

which.max (hatvalues (lm.fit))



#----------------------------------------------------#
# multiple linear regression

lm.fit = lm(medv ~ lstat+age, data = Boston)
summary (lm.fit)

lm.fit = lm(medv~., data = Boston) # all features
summary(lm.fit)


library(car)
vif(lm.fit) # comppute variance inflation factor

lm.fit1=lm(medv~.-age ,data=Boston)
summary (lm.fit1)

lm.fit1=update (lm.fit , ~.-age)
summary(lm.fit1)


#----------------------------------------------------#
# Interaction Terms

summary(lm(medv~lstat * age ,data=Boston ))

#----------------------------------------------------#
# Non-linear Transformations of the Predictors

lm.fit2=lm(medv~lstat + I(lstat^2))
summary(lm.fit2)

lm.fit =lm(medv~lstat)
anova(lm.fit ,lm.fit2)

par(mfrow=c(2,2))
plot(lm.fit2)

lm.fit5=lm(medv~poly(lstat ,5)) # produces a fifth-order polynomial fit
summary (lm.fit5) 

summary(lm(medv~log(rm),data=Boston )) # log transformation


#----------------------------------------------------#
# Qualitative Predictors

fix( Carseats )
names(Carseats )

lm.fit =lm(Sales~.+ Income:Advertising + Price:Age ,data=Carseats )
summary(lm.fit)

attach (Carseats )
contrasts (ShelveLoc )

#----------------------------------------------------#
# Writing Functions

LoadLibraries = function (){
    library (ISLR)
    library (MASS)
    print (" The libraries have been loaded .")
}

LoadLibraries()
