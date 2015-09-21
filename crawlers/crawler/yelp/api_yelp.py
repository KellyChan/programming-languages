"""
Author: Kelly Chan
Date: July 12 2014
"""

import time
import rauth

import pandas
import simplejson as json


def defineParams(latitude, longitude):
    params = {}
    params["term"] = "restaurants"
    params["ll"] = "{},{}".format(str(latitude), str(longitude))
    params["radius_filter"] = "2000"
    params["limit"] = "10"
 
    return params

def getData(params):

    # setting up personal Yelp account
    consumer_key = "XXX"
    consumer_secret = "XXXX"
    token = "XXX"
    token_secret = "XXX"
   
    session = rauth.OAuth1Session(consumer_key = consumer_key,
                                  consumer_secret = consumer_secret,
                                  access_token = token,
                                  access_token_secret = token_secret)
     
    request = session.get("http://api.yelp.com/v2/search", params=params)
   
    # transforming the data in JSON format
    data = request.json()
    session.close()

    return data

def main():

    locations = [(39.98,-82.98),(42.24,-83.61),(41.33,-89.13)]
    
    apiData = []
    for latitude, longitude in locations:
        params = defineParams(latitude, longitude)
        apiData.append(getData(params))
        time.sleep(1.0)

    #print len(apiData)

    for key in apiData[0].keys():
        print key

    for record in apiData:
        print record["businesses"]
        print record['total']
        print record['region']
    print(json.dumps(apiData, sort_keys=True, indent=4 * ' '))




if __name__ == '__main__':
    main()
