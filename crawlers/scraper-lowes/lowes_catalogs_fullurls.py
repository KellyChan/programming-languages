# -*- coding: utf-8 -*-  

"""
Project: eCatalog - Lowes
- link: http://www.lowes.com/

Author: Kelly Chan
Date: Sept 4 2014

Version: v1.0.0
"""

import os
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import re
import time
import urllib
import urllib2
from bs4 import BeautifulSoup

import pandas

def getHTML(url):

    #time.sleep(1.00)
    
    headers = {'User-Agent' : "Mozilla/5.0 (Windows NT 6.0; WOW64) AppleWebKit/534.24 (KHTML, like Gecko) Chrome/11.0.696.16 Safari/534.24"}
    req = urllib2.Request(url,None,headers)
    html = urllib2.urlopen(req).read()
    urllib2.urlopen(url).close()

    soup = BeautifulSoup(html)

    return soup

def extractAttribute(contents, pattern):
    return re.findall(re.compile(pattern), str(contents))

def extractURL(soup):

    results = soup.find('ul', attrs={'class': 'categories sep'})
    #print results

    # links
    pattern = r"""<a href="(.*) title=.*>"""
    links = extractAttribute(results, pattern)
    #print links

    return links


def extract(url):

    soup = getHTML(url)
    data = extractURL(soup)

    return data

def dfs(links, baseurl, fileName):
    
    for link in links:
        thisLink = link.replace('"', '')
        thisURL = baseurl + thisLink
        newURLs = extract(thisURL)
        if newURLs:
            dfs(newURLs, baseurl, fileName)
        else:
            outCSV(fileName, thisLink)
    

def outCSV(fileName, url):

    with open(fileName, 'a') as f:
        f.write("%s\n" % url)

def main():

    baseurl = "http://www.lowes.com"
    url = "http://www.lowes.com/Hidden-Catalogs/_/N-1z138z4/pl#!"
    outPath = "G:/vimFiles/freelance/20140903-eCatalog/src/outputs/"
    fullurls = "lowes-catalogs-fullurls.csv"

    if os.path.exists(outPath+fullurls):
        os.remove(outPath+cateName)

    links = extract(url)
    dfs(links, baseurl, outPath+fullurls)

    #with open(outPath+fullurls, 'wb') as f:
    #    for line in data:
    #        f.write("%s\n" % line)
    #f.close()


if __name__ == "__main__":
    main()

