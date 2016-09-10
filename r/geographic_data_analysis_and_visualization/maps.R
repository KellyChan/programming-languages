#
# Author: Kelly Chan
# Date  : Aug 8 2013
#
# Project: Geographic Data Analysis and Visualization
# Source: http://geography.uoregon.edu/GeogR/topics/index.html
#
# example: Maps in R -- Examples
# source: http://geography.uoregon.edu/GeogR/topics/maps.htm
#
#--------------------------------------------------------------------------#
# Set Work Directory / Load data

setwd("G:\\eclipseWorkspace\\R\\examples\\geographicDataAnalysisAndVisualization")

#library(gpclib)
library(maptools)     # loads sp library too
library(RColorBrewer) # creates nice color schemes
library(classInt)     # finds class intervals for continuous variables

gpclibPermit()

# Read the shapefiles using the maptools function read.shape()

# outlines of Oregon counties (lines)
# browse to orotl.shp
orotl.shp <- readShapeLines("data\\orotl.shp",
		                    proj4string=CRS("+proj=longlat"))

# Oregon climate station data (points)
# browse to orstations.shp
orstations.shp <- readShapePoints("data\\orstations.shp",
		                          proj4string=CRS("+proj=longlat"))

# Oregon county census data (polygons)
# browse to orcounty.shp
orcounty.shp <- readShapePoly("data\\orcounty.shp",
		                      proj4string=CRS("+proj=longlat"))
					  
# Read ordinary rectangular data sets:

orstationc <- read.csv("data\\orstationc.csv")
orcountyp <- read.csv("data\\orcountyp.csv")
cities <- read.csv("data\\cities.csv")
cities2 <- read.csv("data\\cities2.csv")

# Examine the structure and contents of orcounty.shp shapefile:

summary(orcounty.shp)
attributes(orcounty.shp)
attributes(orcounty.shp@data)
attr(orcounty.shp,"polygons")

#--------------------------------------------------------------------------#
# Simple Maps
#---#

#R has the capability of plotting some simple maps using the maptools package, 
#which can read and plot ESRI shapefiles.  

#--------------------------------------------------------------------------#
# Oregon county census data -- attribute data in the orcounty.shp shape file

# equal-frequency class intervals
plotvar <- orcounty.shp@data$POP1990
nclr    <- 8
plotclr <- brewer.pal(nclr,"BuPu")
class   <- classIntervals(plotvar, nclr, style="quantile")
colcode <- findColours(class, plotclr)


plot(orcounty.shp, xlim=c(-124.5, -115), ylim=c(42,47))
plot(orcounty.shp, col=colcode, add=T)
title(main="Population 1990",
	  sub="Quantile (Equal-Frequency) Class Intervals")
legend(-117, 44, legend=names(attr(colcode, "table")),
		         fill=attr(colcode, "palette"), 
				 cex=0.6, bty="n")

#--------------------------------------------------------------------------#
# Oregon climate station data
# data in the orstationc.csv file, basemap in orotl.shp

# symbol plot -- equal-interval class intervals
plotvar <- orstationc$tann
nclr    <- 8
plotclr <- brewer.pal(nclr,"PuOr")
plotclr <- plotclr[nclr:1] # reorder colors
class   <- classIntervals(plotvar, nclr, style="equal")
colcode <- findColours(class, plotclr)


plot(orotl.shp, xlim=c(-124.5, -115), ylim=c(42,47))
points(orstationc$lon, orstationc$lat, pch=16, col=colcode, cex=2)
points(orstationc$lon, orstationc$lat, cex=2)
title("Oregon Climate Station Data -- Annual Temperature",
	  sub="Equal-Interval Class Intervals")
legend(-117, 44, legend=names(attr(colcode, "table")),
		         fill=attr(colcode, "palette"), 
				 cex=0.6, bty="n")

#--------------------------------------------------------------------------#
# Oregon climate station data -- locations and data in shape file

# symbol plot -- equal-interval class intervals
plotvar <- orstations.shp@data$pann
nclr    <- 5
plotclr <- brewer.pal(nclr,"BuPu")
class   <- classIntervals(plotvar, nclr, 
		                  style="fixed",
		                  fixedBreaks=c(0,200,500,1000,2000,5000))
colcode <- findColours(class, plotclr)
orstations.pts <- orstations.shp@coords # get point data


plot(orotl.shp, xlim=c(-124.5, -115), ylim=c(42,47))
points(orstations.pts, pch=16, col=colcode, cex=2)
points(orstations.pts, cex=2)
title("Oregon Climate Station Data -- Annual Precipitation",
	  sub="Fixed-Interval Class Intervals")
legend(-117, 44, legend=names(attr(colcode, "table")),
		         fill=attr(colcode, "palette"), cex=0.6, bty="n")		 
		 
#--------------------------------------------------------------------------#
# Variations in color scales and representation
#---#		 

# This set of examples illustrates some more applications of the maptools package, 
# and some variations in the contruction of class intervals 
# for choropleth maps and symbolic representation 
# of the Oregon county-level census data

#--------------------------------------------------------------------------#
# Oregon county census data -- equal-frequency class intervals

# equal-frequency class intervals
plotvar  <- orcounty.shp@data$AREA
nclr     <- 8
plotclr  <- brewer.pal(nclr,"BuPu")
#plotclr <- plotclr[nclr:1] # reorder colors if appropriate
class    <- classIntervals(plotvar, nclr, style="quantile")
colcode  <- findColours(class, plotclr)


plot(orcounty.shp, xlim=c(-124.5, -115), ylim=c(42,47))
plot(orcounty.shp, col=colcode, add=T)
title(main="Area",
	  sub="Quantile (Equal-Frequency) Class Intervals")
legend(-117, 44, legend=names(attr(colcode, "table")),
		         fill=attr(colcode, "palette"), 
				 cex=0.6, bty="n")


#--------------------------------------------------------------------------#
# Oregon county census data -- equal-width class intervals

# equal-width class intervals of county area
plotvar  <- orcounty.shp@data$AREA
nclr     <- 8
plotclr  <- brewer.pal(nclr,"BuPu")
#plotclr <- plotclr[nclr:1] # reorder colors if appropriate
class    <- classIntervals(plotvar, nclr, style="equal")
colcode  <- findColours(class, plotclr)


plot(orcounty.shp, xlim=c(-124.5, -115), ylim=c(42,47))
plot(orcounty.shp, col=colcode, add=T)
title(main="Area",
	  sub=" Equal-Width Class Intervals")
legend(-117, 44, legend=names(attr(colcode, "table")),
		         fill=attr(colcode, "palette"), 
				 cex=0.6, bty="n")


#equal-width class intervals of 1990 population
plotvar  <- orcounty.shp@data$POP1990
nclr     <- 8
plotclr  <- brewer.pal(nclr,"BuPu")
#plotclr <- plotclr[nclr:1] # reorder colors if appropriate
class    <- classIntervals(plotvar, nclr, style="equal")
colcode  <- findColours(class, plotclr)


plot(orcounty.shp, xlim=c(-124.5, -115), ylim=c(42,47))
plot(orcounty.shp, col=colcode, add=T)
title(main="Population 1990",
	  sub=" Equal-Width Class Intervals")
legend(-117, 44, legend=names(attr(colcode, "table")),
		         fill=attr(colcode, "palette"), cex=0.6, bty="n")


#--------------------------------------------------------------------------#
# Oregon county census data -- bubble plots

# bubble plot equal-frequency class intervals
plotvar  <- orcounty.shp@data$AREA
nclr     <- 8
plotclr  <- brewer.pal(nclr,"BuPu")
#plotclr <- plotclr[nclr:1] # reorder colors if appropriate
max.symbol.size=12
min.symbol.size=1
class    <- classIntervals(plotvar, nclr, style="quantile")
colcode  <- findColours(class, plotclr)
symbol.size <- ((plotvar-min(plotvar))/(max(plotvar)-min(plotvar))*
			    (max.symbol.size-min.symbol.size)+
			    min.symbol.size
			   )

lot(orcounty.shp, xlim=c(-124.5, -115), ylim=c(42,47))
orcounty.cntr <- coordinates(orcounty.shp)
points(orcounty.cntr, pch=16, col=colcode, cex=symbol.size)
points(orcounty.cntr, cex=symbol.size)
text(-120, 46.5, "Area: Equal-Frequency Class Intervals")
legend(locator(1), legend=names(attr(colcode, "table")),
		           fill=attr(colcode, "palette"), 
				   cex=0.6, bty="n")


# bubble plot equal-frequency class intervals
plotvar <- orcounty.shp@data$POP1990
nclr    <- 8
plotclr <- brewer.pal(nclr,"BuPu")
plotclr <- plotclr[nclr:1] # reorder colors if appropriate
max.symbol.size=12
min.symbol.size=1
class   <- classIntervals(plotvar, nclr, style="quantile")
colcode <- findColours(class, plotclr)
symbol.size <- ((plotvar-min(plotvar))/(max(plotvar)-min(plotvar))*
			    (max.symbol.size-min.symbol.size)+
				min.symbol.size
			   )


plot(orcounty.shp, xlim=c(-124.5, -115), ylim=c(42,47))
orcounty.cntr <- coordinates(orcounty.shp)
points(orcounty.cntr, pch=16, col=colcode, cex=symbol.size)
points(orcounty.cntr, cex=symbol.size)
text(-120, 46.5, "Population: Equal-Frequency Class Intervals")
legend(locator(1), legend=names(attr(colcode, "table")),
		           fill=attr(colcode, "palette"), 
				   cex=0.6, bty="n")


#--------------------------------------------------------------------------#
# Oregon county census data -- (pseudo) dot-density maps

# maptools dot-density maps
# warning: this can take a little while
plottitle <- "Population 1990, each dot=1000"
orpolys   <- SpatialPolygonsDataFrame(orcounty.shp, data=as(orcounty.shp, "data.frame"))
plotvar   <- orpolys@data$POP1990/1000.0


dots.rand <- dotsInPolys(orpolys, as.integer(plotvar), f="random")
plot(orpolys, xlim=c(-124.5, -115), ylim=c(42,47))
plot(dots.rand, add=T, pch=19, cex=0.5, col="magenta")
plot(orpolys, add=T)
title(plottitle)



dots.reg <- dotsInPolys(orpolys, as.integer(plotvar), f="regular")
plot(orpolys, xlim=c(-124.5, -115), ylim=c(42,47))
plot(dots.reg, add=T, pch=19, cex=0.5, col="purple")
plot(orpolys, add=T)
title(plottitle)

#--------------------------------------------------------------------------#
# Projections
#---#	

library(rgdal)

# Oregon climate station data -- data in the orstations shapefile

# equal-frequency class intervals -- spplot & projected
plotvar <- orstations.shp@data$tann # gets data from shapefile .dbf
nclr    <- 8
plotclr <- brewer.pal(nclr,"PuOr")
plotclr <- plotclr[nclr:1] # reorder colors
class   <- classIntervals(plotvar, nclr, style="quantile")
colcode <- findColours(class, plotclr)

# add proj4string info to shape files
proj4string(orotl.shp)      <- CRS("+proj=longlat")
proj4string(orstations.shp) <- CRS("+proj=longlat")

# do the projection
# Albers equal-area projection parameters
aea.proj <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-110
		     +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m"
# project the county outlines and station locations
orotl.shp.proj <- spTransform(orotl.shp, CRS(aea.proj))
orstations.shp.proj <- spTransform(orstations.shp, CRS(aea.proj))

# generate and project a graticule
orgrat <- gridlines(orotl.shp, easts=seq(-127,-115,by=1),
		                       norths=seq(42,47,by=1), 
							   ndiscr = 20)
orgrat.proj <- spTransform(orgrat, CRS(aea.proj))

# specify a state and county outline file and graticule
basemap1 <- list("sp.lines", orgrat.proj, col="gray", fill=NA)
basemap2 <- list("sp.lines", orotl.shp.proj, fill=NA)

# finally, plot the map


spplot(orstations.shp.proj, "tann", col.regions=plotclr, 
		                            cuts=nclr, 
									at=round(class$brks, digits=1),
		                            xlim=bbox(orotl.shp.proj)[1,], 
									ylim=bbox(orotl.shp.proj)[2,],
		                            key.space="right",
		                            sp.layout=list(basemap1, basemap2),
		                            main="Annual Temperature")


#--------------------------------------------------------------------------#
# Oregon county census data -- attribute data in the orcounty.shp shape file

# project Oregon county data
aea.proj <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-110
		     +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m"
orcounty.shp.proj <- spTransform(orcounty.shp, CRS(aea.proj))

# equal-frequency class intervals -- spplot & projected
plotvar <- orcounty.shp@data$AREA
nclr    <- 8
plotclr <- brewer.pal(nclr,"BuPu")
class   <- classIntervals(plotvar, nclr, style="equal")
colcode <- findColours(class, plotclr)


spplot(orcounty.shp.proj, "AREA", col.regions=plotclr, 
		                          at=round(class$brks, digits=1))


# equal-width class intervals -- spplot & projected
plotvar <- orcounty.shp@data$AREA
nclr    <- 8
plotclr <- brewer.pal(nclr,"BuPu")
class   <- classIntervals(plotvar, nclr, style="quantile")
colcode <- findColours(class, plotclr)


spplot(orcounty.shp.proj, "AREA", col.regions=plotclr, 
		                          at=round(class$brks, digits=1))

#--------------------------------------------------------------------------#
# Examples using the maps package
#---#	

# Oregon county census data -- bubble plots

library(maps)
library(mapproj)
# look at the orcounty.csv data
attach(orcountyp)
summary(orcountyp)

# check names for proper order
map("county","oregon", names=T, plot=F)
print(orcountyp$Name)


# bubble plot equal-frequency class intervals
map("county", "oregon", xlim=c(-125,-114), ylim=c(42,47))
map.axes()
plotvar <- orcountyp$Area
nclr    <- 8
plotclr <- brewer.pal(nclr,"BuPu")
#plotclr <- plotclr[nclr:1] # reorder colors if appropriate
max.symbol.size=10
min.symbol.size=1
class   <- classIntervals(plotvar, nclr, style="quantile")
colcode <- findColours(class, plotclr)
symbol.size <- ((plotvar-min(plotvar))/(max(plotvar)-min(plotvar))*
			    (max.symbol.size-min.symbol.size)+
				min.symbol.size)
	
points(orcountyp$Longitude, orcountyp$Latitude, pch=16, 
		                                        col=colcode,
		                                        cex=symbol.size)
points(orcountyp$Longitude, orcountyp$Latitude, cex=symbol.size)
text(-120, 46.5, "Equal-Frequency Class Intervals")
legend(locator(1), legend=names(attr(colcode, "table")),
		           fill=attr(colcode, "palette"), 
				   cex=0.6, bty="n")

#--------------------------------------------------------------------------#
# Basemaps using the maps package
#---#							


# map of large cities
data(world.cities) # make the world cities location data set from the maps package available

# match the large cities with those in the database
m <- match(paste(tolower(as.character(cities2$City)),
				 tolower(as.character(cities2$Country))),
		   paste(tolower(world.cities$name),
				 tolower(world.cities$country.etc)))

 # assign the world.cities location information to the large cities
big.cities      <- NULL
big.cities$name <- cities2$City
big.cities$long <- world.cities$long[m]
big.cities$lat  <- world.cities$lat[m]
big.cities


# plot the map

map("world")
map.axes()
points(big.cities$long,
	   big.cities$lat, 
	   col="blue")
text(big.cities$long, 
	 big.cities$lat, 
	 big.cities$name, 
	 col="red", 
	 cex=.5)



#--------------------------------------------------------------------------#
# Projected maps of large cities

# map of large cities
m <- match(paste(tolower(as.character(cities2$City)),
				 tolower(as.character (cities2$Country))),
		   paste(tolower(world.cities$name),
				 tolower(world.cities$country.etc)))
big.cities      <- NULL
big.cities$name <- cities$City
big.cities$long <- world.cities$long[m]
big.cities$lat  <- world.cities$lat[m]

# map projection information
proj.type <- "azequalarea"
proj.orient <- c(90,0,30)

# plot the map

map("world", proj=proj.type, 
		     orient=proj.orient, 
			 resolution=0, 
			 wrap=T)
map.grid(col="black", labels=F, lty=1)
proj.coords <- mapproject(big.cities$long,
		                  big.cities$lat, 
						  proj=proj.type, 
						  orient=proj.orient)
points(proj.coords, col="blue")
text(proj.coords, labels=big.cities$name, 
		          col="red", 
				  cex=1.25)

	  
#--------------------------------------------------------------------------#
# Further examples illustrating map projection using the maps package
#---#			  

# Unprojected maps

# unprojected

map("county", "oregon", fill=F)
points(orstationc$lon, orstationc$lat)
text(orstationc$lon, orstationc$lat, labels=orstationc$station, 
	                                 col="red",
	                                 cex=.8)


# unprojected, add surrounding states

map("county", c("oregon","washington", "california", "nevada", "idaho"), 
		      xlim=c(-125,-116), 
			  ylim=c(41,47), 
			  fill=F)
map("state", "oregon", fill=F, 
		               col="grey", 
					   lwd=3, 
					   add=T)
points(orstationc$lon, orstationc$lat)
text(orstationc$lon, orstationc$lat, labels=orstationc$station, 
		                             col="red",
		                             cex=.8)


# Projected maps

# projected
proj.type    <- "albers"
proj.stdlats <- c(29.5, 45.5)
proj.orient  <- c(90,-120,0)
map("county", c("oregon","washington"), proj=proj.type, 
		                                par=proj.stdlats, 
										orient=proj.orient, 
										fill=F)
orstationc.xy <- mapproject(orstationc$lon, orstationc$lat, proj=proj.type,
		                                                    orient=proj.orient, 
															par=proj.stdlats)

												
map("state", "oregon", proj=proj.type, 
		               par=proj.stdlats,
		               orient=proj.orient, 
					   fill=F, 
					   col="grey", 
					   lwd=3, 
					   add=T)
points(orstationc.xy)
text(orstationc.xy, labels=orstationc$station, 
		            col="red", 
					cex=.8)

			
# projected, whole US
proj.type    <- "albers"
proj.stdlats <- c(29.5, 45.5)
proj.orient  <- c(90,-100,0)


map("world", c("canada", "mexico"), proj=proj.type, 
		                            par=proj.stdlats, 
									orient=proj.orient, 
									xlim=c(-130, -50), 
									ylim=c(25,55), 
									resolution=0, 
									wrap=T)
map("state", proj=proj.type, 
		     par=proj.stdlats, 
			 orient=proj.orient, 
			 resolution=0, 
			 wrap=T, 
			 add=T)
	 
map.grid(col="gray", labels=F, lty=2)
map("state", "oregon", proj=proj.type, 
		               par=proj.stdlats,
		               orient=proj.orient, 
					   fill=F, 
					   col="grey", 
					   lwd=3, 
					   add=T)
orstationc.xy <- mapproject(orstationc$lon, orstationc$lat, proj=proj.type, 
		                                                    par=proj.stdlats, 
															orient=proj.orient)
points(orstationc.xy, col="blue")


#--------------------------------------------------------------------------#
# Basemap shapefile generation using the maps package
#---#	

# The map2SpatialLines() function in the maptools function 
# can be used to transform lines extracted from the maps package 
# into sp() package-compatible format 
# to provide basemaps for plotting other data sets. 
# Here is an example for the Pacific Northwest.

# Extraction of a Pacific Northwest basemap

#library(maptools) # also loads sp package
#library(maps)

# extract county outlines from maps() database
pnw.outlines <- map("county", c("oregon","washington", "california",
				                "nevada", "idaho"),
		                      xlim=c(-124.5, -116.0), 
							  ylim=c(41.0, 47.0))

# prune the lines to Washington, Oregon, and Northern California extent
pnw.outlines <- pruneMap(pnw.outlines, xlim=c(-125.0, -115.0), 
		                               ylim=c(41.0,47.0))

# convert to sp lines
pnw.outlines.sp <- map2SpatialLines(pnw.outlines, proj4string=CRS("+proj=longlat"))


plot(pnw.outlines.sp, col="gray", lwd=2)
degAxis(1, at=seq(-125.,-116., by=1.))
degAxis(2, at=seq(42.,47., by=1.))


# project the outlines
aea.proj <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5
		     +lon_0=-120 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m"
pnw.outlines.proj <- spTransform(pnw.outlines.sp, CRS(aea.proj))

# generate and project gridlines
grat <- gridlines(pnw.outlines.sp, easts=seq(-127,-115,by=1),
		                           norths=seq(40,47,by=1), 
								   ndiscr = 20)
grat.proj <- spTransform(grat, CRS(aea.proj))


plot(pnw.outlines.proj, col="gray", lwd=2)
lines(grat.proj, col="blue", lty=3)
