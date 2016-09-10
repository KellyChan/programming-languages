"""
Project: Data Science
Subject: SQL/Python - 4. orders

Author: Kelly Chan
Date: May 8 2014
"""

import pandas
import MySQLdb

def connectMySQL(host, user, passwd, db, sql):

    try:
        conn = MySQLdb.connect(host, user, passwd, db)
        cursor = conn.cursor()

        n = cursor.execute(sql)
        data = cursor.fetchall()

        cursor.close()
        conn.close()

        return data

    except Exception, e:
        print "MySQL server could not be connected."

def createDataFrame(data):

    codes = []
    cates = []
    for row in data:
        code, cate = row
        codes.append(code)
        cates.append(cate)

    dataDF = pandas.DataFrame({'codes': codes, \
                               'cates': cates
                              })
    return dataDF


def main():
    data = connectMySQL('localhost', 'root', '', 'ds2', \
                        'SELECT * from categories')
    dataDF = createDataFrame(data)
    print type(dataDF)
    print dataDF

if __name__ == '__main__':
    main()
