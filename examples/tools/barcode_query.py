#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Kelly Chan'
__date__ = 'Oct 1 2014'
__version__ = '1.0.0'


import sys 
reload(sys) 
sys.setdefaultencoding('utf8')

import json

import urllib
import urllib2
import httplib


def ConnectDatabase(barcode):

    url = 'http://setup.3533.com/ean/index?keyword=' + str(barcode)
    connection = httplib.HTTPConnection("setup.3533.com")
    connection.request(method='GET', url=url)

    response = connection.getresponse().read()
    info = json.loads(response)

    return info

def PrinInfo(info):

    print "code:     " + info.get("ean","null")
    print "name:     " + info.get("name","null")
    print "price:    " + str(info.get("price","null"))
    print "supplier: " + info.get("supplier","null")
    print "factory:  " + info.get("production","null")
    print ""

if __name__ == '__main__':

    info = ConnectDatabase(6939354800469)
    PrinInfo(info)

    info = ConnectDatabase(6917878002972)
    PrinInfo(info)

    info = ConnectDatabase(6925785604585)
    PrinInfo(info)


