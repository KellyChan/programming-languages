"""
Project: Data Science
Subject: Machine Learning - SEM

Author: Kelly Chan
Date: May 10 2014
"""

dataPath = "path/sources/"
tabPath = "path/outputs/sem/tables/"
picPath = "path/outputs/sem/pics/"

import pandas as pd
from ggplot import *

def loadData(datafile):

    data = []
    for line in open(datafile, 'rb').readlines():
        thisLine = line.strip().split('\t')
        if len(thisLine) != 15:
            continue
        else:
            data.append(thisLine)

    return data


def parseData(data):

    pdDict = {}
    for item in data[0]:
        pdDict[item] = []

    for row in data[1:]:
        for i in range(15):
            pdDict[data[0][i]].append(row[i])

    data = pd.DataFrame(pdDict)
    return data

def cleanNA(data):
    return data.replace('-', '-1')

def parseConentAdgroup(data):

    N = len(data)
    contents = [x for x in range(N)] 
    adgroups = [x for x in range(N)]

    for index, value in enumerate(data['Content Adgroup']):
        if value == "Sitelinks":
            contents[index] = "Sitelinks"
            adgroups[index] = "NA"
        else:
            contents[index], adgroups[index] = value.split(': ')

    data['Content'] = pd.Series(contents)
    data['Adgroup'] = pd.Series(adgroups)

    return data

def parseContent(data):

    N = len(data)
    content1 = [x for x in range(N)]
    content2 = [x for x in range(N)]
    content3 = [x for x in range(N)]

    data['Content'] = data['Content'].str.replace('[', '')
    data['Content'] = data['Content'].str.replace(']', '')

    for index, value in enumerate(data['Content']):
        value = value.split('|')
        if len(value) == 1:
            content1[index] = value[0]
            content2[index] = "NA"
            content3[index] = "NA"
        elif len(value) == 2:
            content1[index] = value[0]
            content2[index] = value[1]
            content3[index] = "NA"
        elif len(value) == 3:
            content1[index] = value[0]
            content2[index] = value[1]
            content3[index] = value[2]

    data['ContentID'] = pd.Series(content1)
    data['ContentType'] = pd.Series(content2)
    data['ContentSite'] = pd.Series(content3)

    return data

def parseAdgroup(data):

    N = len(data)
    groups = [x for x in range(N)]
    ways = [x for x in range(N)]

    data['Adgroup'] = data['Adgroup'].str.replace(" {ID}", "")

    for index, value in enumerate(data['Adgroup']):
        value = value.split(' (')
        if len(value) == 1:
            groups[index] = value[0]
            ways[index] = "NA"
        else:
            groups[index] = value[0]
            ways[index] = value[1].replace(")", "").replace(" ", "")

    data['Adgroup Name'] = pd.Series(groups)
    data['Adgroup Type'] = pd.Series(ways)

    return data

def parseKeyword(data):

    N = len(data)
    keywords = [x for x in range(N)]
    types = [x for x in range(N)]

    for index, value in enumerate(data['SEM Keyword']):
        value = value.split('(')

        if len(value) == 2:
            keyword, way = value
            keywords[index] = keyword.replace('+', '').lower()
            types[index] = way.replace(')', '')
        else:
            keywords[index] = 'NA'
            types[index] = 'NA'

    data['Keyword'] = pd.Series(keywords)
    data['KeywordType'] = pd.Series(types)

    return data


def cleanOrderValue(data):

    data['Order Value'] = data['Order Value'].str.replace(",", "")
    data['Order Value'] = data['Order Value'].astype(float)

    return data

def renameCols(data):

    colNames = {'Content/Adgroup': 'Content Adgroup', \
                'Order Value (Attribution multiple, external)': 'Order Value', \
                'OP: Visits w Catalogue %': 'OP Visits w Catalogue %', \
                'OP: Visits w Search %': 'OP Visits w Search %', \
                'OP: Visits w Product %': 'OP Visits w Product %', \
                'OP: Visits w step Cart %': 'OP Visits w step Cart %'}

    data = data.rename(columns=colNames)

    return data

def reorderCols(data):

    return data.reindex(columns=['Hours', \
                                 'Content Adgroup', \
                                 'Content', 'ContentID', 'ContentType', 'ContentSite', \
                                 'Adgroup', 'Adgroup Name', 'Adgroup Type', \
                                 'SEM Keyword', 'Keyword', 'KeywordType', \
                                 'Campaign Clicks', 'Order Value', \
                                 'Visitors', 'New Visitors', \
                                 'Bounces', 'Entry Rate %', 'Exit Rate %', \
                                 'OP Visits w Catalogue %', \
                                 'OP Visits w Search %', \
                                 'OP Visits w Product %', \
                                 'OP Visits w step Cart %', \
                                 'Campaign Lifecycle Contacts'])



def tabSummary(data):
    tab = data.describe()
    tab.to_csv(tabPath + 'summary.csv')
    return tab

def tabFreq(data, col):
    tab = data[col].value_counts().reset_index()
    tab.columns = ['value', 'freq']
    
    tab['percent'] = tab['freq'] / sum(tab['freq'])

    tab.to_csv("%sholecount-%s.csv" % (tabPath, col))
    
    return tab    

def tabCross(rows, cols, tabName):
    
    tab = pd.crosstab(rows=rows, cols=cols, margins=True)
    tab.to_csv("%scrosstab-%s.csv" % (tabPath, tabName))
    return tab 

def plotHist(data, x):

    p = ggplot(aes(x=x), data=data)
    p = p + geom_histogram()
    p = p + ggtitle("Histogram-%s" % str(x))
    ggsave(p, "%sHistogram-%s.png" % (picPath, str(x)))

def plotBar(data, x):

    p = ggplot(aes(x=x), data=data)
    p = p + geom_bar()
    p = p + ggtitle("BarPlot-%s" % str(x))
    ggsave(p, "%sBarPlot-%s.png" % (picPath, str(x)))

def tabAll(data):

    tabSummary(data)

    for col in data.columns:
        tabFreq(data, col)

    cols = ['Hours', \
            'Content Adgroup', 'Content', 'Adgroup', \
            'ContentID', 'ContentType', 'ContentSite', \
            'Campaign Clicks', 'Order Value', \
            'Visitors', 'New Visitors', \
            'Bounces', 'Entry Rate %', 'Exit Rate %', \
            'OP Visits w Catalogue %', \
            'OP Visits w Search %', \
            'OP Visits w Product %', \
            'OP Visits w step Cart %', \
            'Campaign Lifecycle Contacts']
    for col in cols:
        tabName = 'Keyword' + col
        tabCross(data['Keyword'], data[col], tabName)

def plotAll(data):

    cols = ['Hours', 'Campaign Clicks', \
            'Visitors', 'New Visitors', 'Bounces',  \
            'Campaign Lifecycle Contacts']
    for col in cols:
        data[col] = data[col].astype(int)
        plotHist(data, col)

    cols = ['Order Value', \
            'Entry Rate %', 'Exit Rate %', \
            'OP Visits w Catalogue %', 'OP Visits w Search %', \
            'OP Visits w Product %', 'OP Visits w step Cart %']
    for col in cols:
        data[col] = data[col].astype(float)
        plotHist(data, col)


    cols = ['ContentID', 'ContentType', 'ContentSite']
    for col in cols:
        plotBar(data, col)



def main():

    data = loadData(dataPath + "Indonesia 6.csv")
    data = parseData(data)

    data = renameCols(data)

    data = cleanNA(data)
    data = parseConentAdgroup(data)
    data = parseContent(data)
    data = parseAdgroup(data)
    data = parseKeyword(data)
    data = cleanOrderValue(data)

    data = reorderCols(data)
    data.to_csv(tabPath + 'clean_data.csv')

    tabAll(data)
    #plotAll(data)



if __name__ == '__main__':
    main()
