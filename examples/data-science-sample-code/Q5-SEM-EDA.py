"""
Project: Data Science 
Subject: Machine Learning - SEM

Author: Kelly Chan
Date: May 10 2014
"""

tabPath = "path/outputs/sem/tables/"
picPath = "path/outputs/sem/pics/"

import pandas as pd
from ggplot import *

def loadData(datafile):
    return pd.read_csv(datafile)

def plotScatter(data, x, y, tabName):

    p = ggplot(aes(x=x, y=y), data=data)
    p = p + geom_point()
    p = p + ggtitle("Scatter Plot - %s vs. %s - %s" % (str(x), str(y), tabName))
    ggsave(p, "%sScatterPlot-%s_vs._%s-%s.png" % (picPath, str(x), str(y), tabName))

def plotRelation(subData, subName):

    plotScatter(subData, 'Campaign Lifecycle Contacts', 'Campaign Clicks', subName)

    plotScatter(subData, 'Bounces', 'Visitors', subName)
    plotScatter(subData, 'Entry Rate %', 'Visitors', subName)
    plotScatter(subData, 'Entry Rate %', 'Visitors', subName)
    plotScatter(subData, 'Exit Rate %', 'Visitors', subName)

    plotScatter(subData, 'OP Visits w Catalogue %', 'Order Value', subName)
    plotScatter(subData, 'OP Visits w Search %', 'Order Value', subName)
    plotScatter(subData, 'OP Visits w Product %', 'Order Value', subName)
    plotScatter(subData, 'OP Visits w step Cart %', 'Order Value', subName)

def tabSummary(data, groups):

    tab = data.groupby(groups).sum()
    
    return tab

def main():

    data = loadData(tabPath + "clean_data.csv")
    
    subData = data.loc[:, ['Keyword', 'Visitors']] 
    tab = tabSummary(subData, ['Keyword'])
    tab.to_csv(tabPath + 'keyword_visits_orders.csv')

    less100 = tab[tab['Visitors'] < 100]
    index = 0.05 * len(less100)
    tab = less100.iloc[:index]
    tab.to_csv(tabPath + 'visitors_lt_100.csv')

    subData = data.loc[:, ['Adgroup Name', 'Keyword', 'Visitors', 'New Visitors', 'Order Value']]
    tab = tabSummary(subData, ['Adgroup Name', 'Keyword'])
    tab.to_csv(tabPath + "Adgroup_Keyword_Visits_Orders.csv")
    print tab

    #plotRelation(data, "All Keywords")

    # NOTE: careful, it will cost a lot of time
    #for value in data['SEM Keyword'].values:
    #    subData = data[data["SEM Keyword"] == value]
    #    plotRelation(subData, str(value))




if __name__ == '__main__':
    main()
