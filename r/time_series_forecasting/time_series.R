
# Loading data

#-----------------------------------------------------------------------#
# kings
#---------#

kings <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)
kings

kingstimeseries <- ts(kings)
kingstimeseries

#-----------------------------------------------------------------------#
# births
#---------#

births <- scan("http://robjhyndman.com/tsdldata/data/nybirths.dat")
births

birthstimeseries <- ts(births, frequency=12, start=c(1946,1))
birthstimeseries

#-----------------------------------------------------------------------#
# souvenir
#---------#

souvenir <- scan("http://robjhyndman.com/tsdldata/data/fancy.dat")
souvenirtimeseries <- ts(souvenir, frequency=12, start=c(1987,1))
souvenirtimeseries

#-----------------------------------------------------------------------#

# Plotting

plot.ts(kingstimeseries)
plot.ts(birthstimeseries)
plot.ts(souvenirtimeseries)

logsouvenirtimeseries <- log(souvenirtimeseries)
plot.ts(logsouvenirtimeseries)

#-----------------------------------------------------------------------#

# Decomposing

#-----------------------------------------------------------------------#
# smoothing non-seasonal data
#---------#

library("TTR")

kingstimeseriesSMA3 <- SMA(kingstimeseries,n=3)
plot.ts(kingstimeseriesSMA3)

kingstimeseriesSMA8 <- SMA(kingstimeseries,n=8)
plot.ts(kingstimeseriesSMA8)

#-----------------------------------------------------------------------#
# smoothing seasonal data
#---------#

birthstimeseriescomponents <- decompose(birthstimeseries)
birthstimeseriescomponents$seasonal 
plot(birthstimeseriescomponents)

# adjusting

birthstimeseriescomponents <- decompose(birthstimeseries)
birthstimeseriesseasonallyadjusted <- birthstimeseries - birthstimeseriescomponents$seasonal
plot(birthstimeseriesseasonallyadjusted)

#-----------------------------------------------------------------------#

# Forecasting


#-----------------------------------------------------------------------#
# Exponential Smoothing
#---------#

rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
rainseries <- ts(rain,start=c(1813))
plot.ts(rainseries)

rainseriesforecasts <- HoltWinters(rainseries, beta=FALSE, gamma=FALSE)
rainseriesforecasts

rainseriesforecasts$fitted
plot(rainseriesforecasts)

rainseriesforecasts$SSE


HoltWinters(rainseries, beta=FALSE, gamma=FALSE, l.start=23.56)

library("forecast")

rainseriesforecasts2 <- forecast.HoltWinters(rainseriesforecasts, h=8)
rainseriesforecasts2
plot.forecast(rainseriesforecasts2)
acf(rainseriesforecasts2$residuals, lag.max=20)

Box.test(rainseriesforecasts2$residuals, lag=20, type="Ljung-Box")
plot.ts(rainseriesforecasts2$residuals)

plotForecastErrors <- function(forecasterrors)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors)/4
  mysd   <- sd(forecasterrors)
  mymin  <- min(forecasterrors) - mysd*5
  mymax  <- max(forecasterrors) + mysd*3
  
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

plotForecastErrors(rainseriesforecasts2$residuals)


#-----------------------------------------------------------------------#
# Holt’s Exponential Smoothing
#---------#

skirts <- scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5)
skirtsseries <- ts(skirts,start=c(1866))
plot.ts(skirtsseries)

skirtsseriesforecasts <- HoltWinters(skirtsseries, gamma=FALSE)
skirtsseriesforecasts
skirtsseriesforecasts$SSE

plot(skirtsseriesforecasts)


HoltWinters(skirtsseries, gamma=FALSE, l.start=608, b.start=9)
skirtsseriesforecasts2 <- forecast.HoltWinters(skirtsseriesforecasts, h=19)
plot.forecast(skirtsseriesforecasts2)

acf(skirtsseriesforecasts2$residuals, lag.max=20)
Box.test(skirtsseriesforecasts2$residuals, lag=20, type="Ljung-Box")


plot.ts(skirtsseriesforecasts2$residuals)    
plotForecastErrors(skirtsseriesforecasts2$residuals) 


#-----------------------------------------------------------------------#
# Holt-Winters Exponential Smoothing
#---------#

logsouvenirtimeseries <- log(souvenirtimeseries)
souvenirtimeseriesforecasts <- HoltWinters(logsouvenirtimeseries)
souvenirtimeseriesforecasts

plot(souvenirtimeseriesforecasts)

souvenirtimeseriesforecasts2 <- forecast.HoltWinters(souvenirtimeseriesforecasts, h=48)
plot.forecast(souvenirtimeseriesforecasts2)

acf(souvenirtimeseriesforecasts2$residuals, lag.max=20)
Box.test(souvenirtimeseriesforecasts2$residuals, lag=20, type="Ljung-Box")

plot.ts(souvenirtimeseriesforecasts2$residuals) 
plotForecastErrors(souvenirtimeseriesforecasts2$residuals)


#-----------------------------------------------------------------------#
# ARIMA Models
#---------#

skirtsseriesdiff1 <- diff(skirtsseries, differences=1)
plot.ts(skirtsseriesdiff1)


skirtsseriesdiff2 <- diff(skirtsseries, differences=2)
plot.ts(skirtsseriesdiff2)


kingtimeseriesdiff1 <- diff(kingstimeseries, differences=1)
plot.ts(kingtimeseriesdiff1)


acf(kingtimeseriesdiff1, lag.max=20)   
acf(kingtimeseriesdiff1, lag.max=20, plot=FALSE)

pacf(kingtimeseriesdiff1, lag.max=20) # plot a partial correlogram
pacf(kingtimeseriesdiff1, lag.max=20, plot=FALSE) # get the partial autocorrelation values



volcanodust <- scan("http://robjhyndman.com/tsdldata/annual/dvi.dat", skip=1)
volcanodustseries <- ts(volcanodust,start=c(1500))
plot.ts(volcanodustseries)

acf(volcanodustseries, lag.max=20)     
acf(volcanodustseries, lag.max=20, plot=FALSE)


pacf(volcanodustseries, lag.max=20)
pacf(volcanodustseries, lag.max=20, plot=FALSE)


kingstimeseriesarima <- arima(kingstimeseries, order=c(0,1,1)) # fit an ARIMA(0,1,1) model
kingstimeseriesarima


library("forecast") # load the "forecast" R library

kingstimeseriesforecasts <- forecast.Arima(kingstimeseriesarima, h=5)
kingstimeseriesforecasts

plot.forecast(kingstimeseriesforecasts)


acf(kingstimeseriesforecasts$residuals, lag.max=20)
Box.test(kingstimeseriesforecasts$residuals, lag=20, type="Ljung-Box")


plot.ts(kingstimeseriesforecasts$residuals)  
plotForecastErrors(kingstimeseriesforecasts$residuals) 
