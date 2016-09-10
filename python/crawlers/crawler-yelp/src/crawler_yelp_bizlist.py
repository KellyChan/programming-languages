"""
Author: Kelly Chan
Date: July 14 2014

Project: Extracting attributes from the html contents (Yelp)

- url: http://www.yelp.com/search?find_desc=%s&find_loc=Los+Angeles&ns=1
- %s: the name of a specific restaurant

steps:
- 1. getting the html contents
- 2. parsing attributes by regexes
- 3. saving the data(.txt) to the local drive

"""

import re
import time
import urllib2
from bs4 import BeautifulSoup

def getHTML(url):
    """ return the html contents from the specific url """

    time.sleep(2.00)
    html = urllib2.urlopen(url,timeout=10).read()
    urllib2.urlopen(url).close()

    soup = BeautifulSoup(html)

    return soup


def extractAttribute(content, pattern):
    """ return the attribute that matching a specific pattern in the content """
    
    return re.findall(re.compile(pattern), str(contents))

def extract(soup, keyIndex):
    """ return all attributes those matching specific patterns in the individual content """

    # getting the individual content
    content = soup.find('div', attrs={'data-key': keyIndex})

    # index
    pattern = r"<span class=\"indexed-biz-name\">(\d+)"
    index = extractAttribute(content, pattern)
    #print index

    # link
    pattern = r"biz-name\".*(/biz/[\w?%?+-?]+)\""
    #pattern = r"biz-name\".*(/biz/.*)\"" 
    link = extractAttribute(content, pattern)
    #print link

    # name
    #pattern = r"biz-name.*>(.*)</a>" 
    pattern = r"biz-name.*>(<.*>.*|\w+\s+<.*>.*)</a>"
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

    # phone
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
    """ outputing the data as txt file """

    with open(outPath+fileName, "wb") as f:
        f.write("index,link,name,rating,review,price,category,neighborhood,address,phone,feedback\n")
        for record in data:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % \
                    (record[0],record[1],record[2],record[3],record[4],record[5],record[6],\
                     record[7],record[8],record[9],record[10]))


def getBizListData(url, bizList):
    """ return the full data set by extracting attributes from the html contents one by one (bizList) """

    data = []
    for restaurant in bizList:
        pageURL = (url % restaurant.replace(" ", "+"))
        soup = getHTML(pageURL)

        record = extract(soup, "1")
        data.append(record)
    
    return data

def main():

    outPath = "path/regexEval/data/"
    url = "http://www.yelp.com/search?find_desc=%s&find_loc=Los+Angeles&ns=1"
    fileName = "bizList_data.txt"

    bizList = ['Chego', \
               'Smitty\'s Famous Fish and Chicken', \
               'Zankou Chicken', \
               'Ambala Dhaba', \
               'Fuddruckers', \
               'Colony Cafe']    
    
    data = getBizListData(url, bizList)
    outTxt(data, outPath, fileName)


if __name__ == "__main__":
    main()