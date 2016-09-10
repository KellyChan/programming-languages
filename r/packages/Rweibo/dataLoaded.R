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
res <- web.search.content("eBay", page = 5, sleepmean = 10, sleepsd = 1)$Weibo
write.csv(res, "output\\res.csv",row.names=TRUE)

#--------------------------------------------------------------------------#
# Account Setting
#---#

# register API in R
registerApp(app_name = "SinaSNA", "3742182308", "45bd00a270dc617dbc03bf22431ad687")
#listApp("SinaSNA")

roauth <- createOAuth(app_name = "SinaSNA", access_name = "rweibo")
#roauth

#roauth$getLimits(TRUE)

#--------------------------------------------------------------------------#
# View
#---#

# 查看公共微博
#res1 <- statuses.public_timeline(roauth, count = 5)
#res1[[1]]

# 查看自己以及关注对象的微博
#res2 <- statuses.friends_timeline(roauth, count = 5)
#res2[[1]]

# 查看双向关注的好友的最新微博
#res3 <- statuses.bilateral_timeline(roauth, count = 5)
#res3[[1]]

# 查看某位微博用户的最新微博
#res4 <- statuses.user_timeline(roauth, screen_name = "rweibo", count = 5)
#res4[[1]]

# 查看当前用户收到的评论
#res5 <- comments.to_me(roauth, count = 5)
#res5[[1]]
