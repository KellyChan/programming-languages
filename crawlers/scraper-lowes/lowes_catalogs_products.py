__author__ = "Kelly Chan"
__date__ = "Sept 8 2014"
__version__ = '1.0.0'

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

def getPages(soup):

    results = soup.find('span', attrs={'class': 'totalPages'})
    if results:
        pages = int(results.get_text())
    else:
        pages = 0 

    return pages

def filterRE(results, pattern):
    return re.findall(re.compile(pattern), str(results))

def getProducts(content):
    products = []
    links = []

    results = content.find_all('h3', attrs={'class': 'productTitle'})
    #print results
    pattern1 = r"<a href=.*>([\d\w\u4e00-\u9fa5\s].*)</a>"
    results1 = filterRE(results, pattern1)
    for result in results1:
        products.append(result.strip())
    
    #print products

    pattern2 = r'<a href="(.*)" name=.*>'
    results2 = filterRE(results, pattern2)
    for result in results2:
        links.append(result.strip())

    #print links

    return products, links

def main():

    baseurl = "http://www.lowes.com"
    outPath = "G:/vimFiles/freelance/20140903-eCatalog/src/outputs/"

    br = openBrowser()
    for i in range(21):
        fileName = "lowes-catalogs-fullurls-%s.csv" % str(i)
        urls = loadURLs(outPath+fileName)

        outCSV = outPath+"products-"+fileName
        if os.path.exists(outCSV):
            os.remove(outCSV)
        
        for url in urls:
            thisURL = baseurl + url
            soup = getSoup(br, thisURL)
            pages = getPages(soup)
            #print soup
            #print pages

            if pages == 0:
                with open(outPath+"error-no-products.csv", 'a') as nf:
                    nf.write('%s\n' % (thisURL))

            elif pages == 1:

                content = soup.find('ul', attrs={'id': 'productResults'})
                products, links = getProducts(content)

                if len(products) == len(links):

                    data = pandas.DataFrame({'product': products, 'prodURL': links})
                    data['deptURL'] = thisURL

                    with open(outCSV, 'a') as f:
                        data.to_csv(f, header=False)
                else:
                    with open(outPath+"error-one-page.csv", 'a') as tf:
                        tf.write("%s\n" % thisURL)

            else:
                
                for page in range(pages):
                    pageURL = thisURL + "?&rpp=32&page=" + str(page+1)
                    thisSoup = getSoup(br, pageURL)
                    content = thisSoup.find('ul', attrs={'id': 'productResults'})
                    products, links = getProducts(content)
                    #print pageURL

                    if len(products) == len(links):

                        data = pandas.DataFrame({'product': products, 'prodURL': links})
                        data['deptURL'] = pageURL

                        with open(outCSV, 'a') as f:
                             data.to_csv(f, header=False)

                    else:
                        with open(outPath+"error-more-pages.csv", 'a') as pf:
                            pf.write('%s\n' % (pageURL))



if __name__ == '__main__':
    main()
