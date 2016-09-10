"""
Author: Kelly Chan
Date: July 18 2014

"""

from pprint import pprint
from factual import Factual
from factual.utils import circle


class Database:

    def __init__(self, key, secret):
        self.key = key
        self.secret = secret
        self.db = Factual(key, secret)


    def schema(self, tableName):
        return self.db.table(tableName).schema()

    def table(self, tableName):
        return self.db.table(tableName)

    def search(self, tableName, searchContent):
        table = self.db.table(tableName)
        tableQuery = table.search(searchContent).include_count(True)
        return tableQuery.included_rows(), \
               tableQuery.total_row_count(), \
               tableQuery.data()

    def filter(self, tableName, condition):
        table = self.db.table(tableName)
        data = table.filters(condition).data()
        return data

    def search_filters(self, tableName, searchContent, condition):
        table = self.db.table(tableName)
        data = table.search(searchContent).filters(condition).data()
        return data

    def search_filters_paging(self, tableName, searchContent, condition, pages):
        table = self.db.table(tableName)
        data = table.search(searchContent).filters(condition).offset(pages).limit(pages).data()
        return data

    def geofilters(self, tableName, searchContent, latitude, longitude, radius):
        table = self.db.table(tableName)
        data = table.search(searchContent).geo(circle(latitude, longitude, radius)).data()
        return data



def main():

    factual = Database('YOUR_KEY', \
                       'YOUR_SECRET')

    schemaPlaces = factual.schema('places')
    #pprint(schemaPlaces)

    places = factual.table('places')
    #print places

    rows, N, data = factual.search('places', 'century city mall')
    #print rows
    #print N
    #pprint(data)

    #  search restaurants (http://developer.factual.com/working-with-categories/)
    data = factual.filter('places', {'category_ids': {'$includes': 347}})
    #pprint(data)

    #  search restaurants or bars
    data = factual.filter('places', {'category_ids': {'$includes_any': [312,347]}})
    #pprint(data)

    #  search entertainment venues but NOT adult entertainment
    filters = {'$and': [ \
                         {'category_ids': {'$includes': 317}}, \
                         {'category_ids': {'$excludes': 318}}  \
                       ]
              }

    data = factual.filter('places', filters)
    #pprint(data)

    #  search for Starbucks in Los Angeles
    tableName = 'places'
    searchContent = 'starbucks'
    filters = {'locality': 'los angeles'}
    data = factual.search_filters(tableName, searchContent, filters)
    #pprint(data)


    #  search for starbucks in Los Angeles or Santa Monica 
    tableName = 'places'
    searchContent = 'starbucks' 
    filters = {'$or':[{'locality':{'$eq':'los angeles'}},{'locality':{'$eq':'santa monica'}}]}
    data = factual.search_filters(tableName, searchContent, filters)
    #pprint(data)

    #  paging 
    tableName = 'places'
    searchContent = 'starbucks' 
    filters = {'$or':[{'locality':{'$eq':'los angeles'}},{'locality':{'$eq':'santa monica'}}]}
    data = factual.search_filters_paging(tableName, searchContent, filters, 20)
    #pprint(data)

    # Geo filter:
    #  coffee near the Factual office
    tableName = 'places'
    searchContent = 'coffee'
    latitude = 34.058583
    longitude = -118.416582
    radius = 1000
    data = factual.geofilters(tableName, searchContent, latitude, longitude, radius)
    pprint(data)

if __name__ == "__main__":
    main()


