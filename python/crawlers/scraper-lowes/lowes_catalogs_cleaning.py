__author__ = "Kelly Chan"
__date__ = "Sept 9 2014"
__version__ = "1.0.0"

import re
import pandas as pd

def loadContent(dataFile):

    with open(dataFile, 'rb') as f:
        content = f.read()
    f.close()

    return content

def filterRE(content, pattern):
    return re.findall(re.compile(pattern), str(content))

def extractAttrs(content):

    prodLinks = []
    prodNames = []
    #ids = []
    itemIDs = []
    cateIDs = []
    modelIDs = []
    prodIDs = []
    depts = []

    # prodLinks
    #pattern = r'(/pd_.*productId=\d+|facetInfo=)'
    pattern = r'(/pd_.*facetInfo=)'
    results = filterRE(content, pattern)
    for result in results:
        prodLinks.append("http://www.lowes.com"+result)

    # names
    pattern = r".*,(.*),http://www.lowes.com/.*"
    results = filterRE(content, pattern)
    for result in results:
        prodNames.append(result)

    # ids 
    #pattern = r"/pd_(.*)_.*__\??"
    #results = filterRE(content, pattern)
    #for result in results:
    #    ids.append(result)

    # itemIDs
    pattern = r"/pd_(\d+)-?"
    results = filterRE(content, pattern)
    for result in results:
        itemIDs.append(result)    

    # cateIDs
    pattern = r"/pd_\d+-(\d+)-?"
    results = filterRE(content, pattern)
    for result in results:
        cateIDs.append(result)

    # modelIDs
    pattern = r"/pd_\d+-\d+-(.*)_.*__\??"
    results = filterRE(content, pattern)
    for result in results:
        modelIDs.append(result.replace('+', ' '))

    # prodIDs
    #pattern = r"productId=(\d+)&amp;"
    pattern = r"productId=(\d+)"
    results = filterRE(content, pattern)
    for result in results:
        prodIDs.append(result)

    # depts
    pattern = r"http://www.lowes.com/(.*)/_/N-.*"
    results = filterRE(content, pattern)
    for result in results:
        depts.append(result.replace('-', ' '))

    cates = []
    for dept in depts:
        cates.append(dept.split('/'))

    #print len(names)
    #print len(itemIDs)
    #print len(cateIDs)
    #print len(modelIDs)
    #print len(prodIDs)
    #print len(cates)

    attrs = pd.DataFrame({'prodURL': prodLinks,
                          'prodName': prodNames,
                          'itemID': itemIDs,
                          'cateID': cateIDs,
                          'modelID': modelIDs,
                          'prodID': prodIDs,
                          'cates': cates})

    return attrs


def main():

    dataPath = "G:/vimFiles/freelance/20140903-eCatalog/data/lowes/products/"
    outPath = "G:/vimFiles/freelance/20140903-eCatalog/src/outputs/"

    for i in range(21):
        dataFile = "products-lowes-catalogs-fullurls-%s.csv" % str(i)

        content = loadContent(dataPath+dataFile)
        attrs = extractAttrs(content)
        attrs.to_csv(outPath+"clean-"+dataFile, header=True, index=False)

    #dataFile = "products-error-one-page.csv"
    #dataFile = "products-error-more-pages.csv"
    #dataFile = "products-error-no-products.csv"

    #content = loadContent(dataPath+dataFile)
    #attrs = extractAttrs(content)
    #attrs.to_csv(outPath+"clean-"+dataFile, header=True, index=False)



if __name__ == '__main__':
    main()
