__author__ = "Kelly Chan"
__date__ = "Sept 9 2014"
__version__ = "1.0.0"


import os
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import mechanize 
import cookielib 

import re
import time
import urllib
import urllib2
from bs4 import BeautifulSoup

import pandas

def openBrowser():

    # Browser 
    br = mechanize.Browser() 
    # Cookie Jar 
    cj = cookielib.LWPCookieJar() 
    br.set_cookiejar(cj) 

    # Browser options 
    br.set_handle_equiv(True) 
    #br.set_handle_gzip(True) 
    br.set_handle_redirect(True) 
    br.set_handle_referer(True) 
    br.set_handle_robots(False) 

    # Follows refresh 0 but not hangs on refresh > 0 
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1) 
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]

    return br

def getSoup(br, url):

    # Open url
    r = br.open(url) 
    html = r.read() 

    soup = BeautifulSoup(html)
    return soup

def loadURLs(dataFile):
    urls = []
    with open(dataFile, 'rb') as f:
        for line in f.readlines():
            urls.append(line.strip())
    return urls


def filterRE(results, pattern):
    return re.findall(re.compile(pattern), str(results))

def main():

    dataPath = "G:/vimFiles/freelance/20140903-eCatalog/data/lowes/"
    outPath = "G:/vimFiles/freelance/20140903-eCatalog/src/outputs/"
    fileName = "error-no-products.csv"

    br = openBrowser()

    products = []
    links = []
    deptLinks = []

    urls = loadURLs(dataPath+fileName)
    for url in urls:
        soup = getSoup(br, url)

        # products
        pattern = r"<title>Shop (.*) at Lowes.com</title>"
        results = filterRE(soup, pattern)
        for result in results:
            products.append(result)

        # links
        pattern = r'"http://www.lowes.com(/pd_.*__\?productId=\d+)'
        results = filterRE(soup, pattern)
        for result in results:
            links.append(result)

        deptLinks.append(url)

    data = pandas.DataFrame({'product': products, 'prodURL': links, 'deptURL': deptLinks})
    data.to_csv(outPath+"products-error-no-products.csv", header=False)


if __name__ == '__main__':
    main()
