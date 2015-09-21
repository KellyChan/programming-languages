"""
Project: eCatalog - Lowe's

Author: Kelly Chan
Date: Sept 5 2014
"""

import os

def loadData(datafile):

    data = []

    with open(datafile, 'rb') as f:
        for line in f.readlines():
            data.append(line)

    return data

def split(data, outPath):

    n = len(data) / 100 + 1
    
    for i in range(n):

        fileName = outPath+"lowes-catalogs-fullurls-"+str(i)+".csv"
        if os.path.exists(fileName):
            os.remove(fileName)

        for line in data[i*100:(i+1)*100]:
            if line:
                with open(fileName, 'ab') as f:
                    f.write("%s" % line)

                

def main():

    outPath = "G:/vimFiles/freelance/20140903-eCatalog/src/outputs/"
    fileName = "lowes-catalogs-fullurls-full.csv"

    data = loadData(outPath+fileName)
    split(data, outPath)
            



if __name__ == '__main__':
    main()
