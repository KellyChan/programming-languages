"""
Author: Kelly Chan
Date: July 14 2014
"""

import re
import time
import urllib2
from bs4 import BeautifulSoup

import csv


def getHTML(url):

    time.sleep(1.00)
    html = urllib2.urlopen(url,timeout=10).read()
    urllib2.urlopen(url).close()

    soup = BeautifulSoup(html)

    return soup


def extractAttribute(contents, pattern):
    return re.findall(re.compile(pattern), str(contents))

def extract(soup, keyIndex):

    content = soup.find('div', attrs={'data-key': keyIndex})

    # index
    pattern = r"<span class=\"indexed-biz-name\">(\d+)"
    index = extractAttribute(content, pattern)
    #print index

    # link
    #pattern = r"biz-name\".*(/biz/[\w?%?+-?]+)\""
    pattern = r"biz-name\".*(/biz/.*)\"" 
    link = extractAttribute(content, pattern)
    #print link

    # name
    pattern = r"biz-name.*>(.*)</a>"
    name = extractAttribute(content, pattern)
    #print name

    # rating
    pattern = r"star-img ([\w+_?]+)"
    rating = extractAttribute(content, pattern)
    #print rating

    # review
    pattern = r"(\d+) reviews"
    review = extractAttribute(content, pattern)
    #print review

    # price
    pattern = r"business-attribute price-range\">(.*)</span>"
    price = extractAttribute(content, pattern)
    #print price

    # category
    pattern = r"cflt=(\w+)"
    category = extractAttribute(content, pattern)
    #print category

    # neighborhood
    pattern = r"neighborhood-str-list\">\n\s+(.*)\s+</span>"
    neighborhood = extractAttribute(content, pattern)
    #print neighborhood

    # address
    #pattern = r"\s+(.*)<br>"
    pattern = r"<address>\n([\s\S]*)</address>"
    address = extractAttribute(content, pattern)
    #print address

    # phones
    #pattern = r"(\([\d]+\) [\d]+-[\d]+)"
    pattern = r"biz-phone\">\n\s+(.*)\n\s+</span>"
    phone = extractAttribute(content, pattern)
    #print phone

    # feedback
    pattern = r"<p class=\"snippet\">(.*)</p>"
    feedback = extractAttribute(content, pattern)
    #print feedback

    return [index, link, name, \
            rating, review, price, category, \
            neighborhood, address, phone, feedback]


def outTxt(data, outPath, fileName):

    with open(outPath+fileName, "wb") as f:
        f.write("index,link,name,rating,review,price,category,neighborhood,address,phone,feedback\n")
        for record in data:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % \
                    (record[0],record[1],record[2],record[3],record[4],record[5],record[6],\
                     record[7],record[8],record[9],record[10]))

def getPagesData(url, pages):

    data = []

    for page in range(pages):
        #print "page: ", page*10
        pageURL = (url % str(page*10))
        #print "pageURL: ", pageURL
        soup = getHTML(pageURL)

        subData = []
        for dataKey in range(page*10+1, page*10+11):
            #print "dataKey: ", dataKey
            record = extract(soup, str(dataKey))
            subData.append(record)
        data.extend(subData)

    return data


def main():

    outPath = "G:/vimFiles/python/interviews/regexEval/data/"

    # getting data from several pages

    pages = 2
    url = "http://www.yelp.com/search?find_loc=los+angeles&cflt=restaurants&start=%s"
    fileName = "raw_pages_data.txt"
    outName = "cleaned_pages_data.tsv"


    data = getPagesData(url, pages)
    outTxt(data, outPath, fileName)




if __name__ == "__main__":
    main()

