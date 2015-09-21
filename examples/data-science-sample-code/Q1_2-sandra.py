"""
Project: Data Science
Subject: Python - 2. Sandra

Author: Kelly Chan
Date: May 7 2014
"""

dataPath = "path/sources/Sandra/"
outPath = "path/outputs/"

import csv

def loadData(dataFile, country, startRow, endRow):
    data = []
    lines = open(dataFile).readlines()[startRow:endRow]
    for line in lines:
        thisLine = line.strip().split("\t")
        thisLine.append(country)  # appending countryCode for each line
        data.append(thisLine)
    return data

def combineData(dataList):
    data = []
    for country in dataList:
        data.extend(country)
    return data

def printLines(data):
    for line in data[:5]:
        print '\t'.join(line)

def writeCSV(data, outfile):
    with open(outfile, 'wb') as f:
        out = csv.writer(f, delimiter=',')
        out.writerow(['Days', 'Page Impressions', 'Visits', 'Bounces', 'Country'])
        out.writerows(data)
    f.close()

def main():
    
    rawID = dataPath + 'webtrekk_report_2012-12-15_Sandra_ID.csv'
    rawPH = dataPath + 'webtrekk_report_2012-12-15_Sandra_PH.csv'
    rawVN = dataPath + 'webtrekk_report_2012-12-15_Sandra_VN.csv'

    dataID = loadData(rawID, 'ID', 1, 9)
    dataPH = loadData(rawPH, 'PH', 1, 10)
    dataVN = loadData(rawVN, 'VN', 9, 18)

    fullData = combineData([dataID, dataPH, dataVN])
    printLines(fullData)

    writeCSV(fullData, outPath + 'answer1.csv')



if __name__ == '__main__':
    main()
