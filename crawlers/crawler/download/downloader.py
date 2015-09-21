"""
Author: Kelly Chan
Date: July 12 2014
"""

import re
import urllib2
from bs4 import BeautifulSoup

def getURLs(url):

    fileURLs = []

    contents = urllib2.urlopen(url).read()
    soup = BeautifulSoup(contents)

    for link in soup.find_all(src=re.compile("/arts/paintings/images/mw")):
        fileURLs.append(link.get('src'))

    return fileURLs


def download(url, outPath, fileName):
    """ downloading files from the web """

    loadedSize = 0
    bufferSize = 8192

    # getting the url of a file
    rawFile = urllib2.urlopen(url)

    # getting the total size of the file
    meta = rawFile.info()
    fileSize = int(meta.getheaders("Content-Length")[0])

    print "File name: %s, Bytes: %s" % (fileName, fileSize)
    print "Downloading..."
    
    f = open(outPath + fileName, 'wb')
    while True:

        # downloading files by blocks
        buffer = rawFile.read(bufferSize)
        if not buffer:
            break
        f.write(buffer)

        # showing the downloading process
        loadedSize += len(buffer)
        status = r"%10d [%3.2f%%]" % (loadedSize, loadedSize * 100. / fileSize)
        print status
    f.close()

    print "%s: Ready!" % (fileName)



def downloadFiles(fileURLs, outPath):

    for url in fileURLs:
        fileName = url.split('/')[-1]
        url = "http://kankanwoo.com/" + url
        download(url, outPath, fileName)


def main():

    url = "http://kankanwoo.com/arts/paintings.asp?c=2"
    outPath = "G:/vimFiles/python/projects/crawler/outputs/img/mountains_and_water/"

    fileURLs = getURLs(url)
    downloadFiles(fileURLs, outPath)


if __name__ == "__main__":
    main()
