"""
Project: Data Science 
Subject: SQL/Python - 5. merging tables

Author: Kelly Chan
Date: May 8 2014
"""

import MySQLdb

class Database:

    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

        self.conn = MySQLdb.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db)
        self.cursor = self.conn.cursor()

    def insert(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except:
            self.conn.rollback()

    def query(self, sql):
        cursor = self.conn.cursor()
        cursor.execute(sql)

        return cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()


def mergeData(dataList):
    fullData = []
    for data in dataList:
        fullData.extend(data)
    return fullData

def main():
    db1 = Database('localhost', 'root', '', 'ds2')
    db2 = Database('localhost', 'root', '', 'ds2')
    db3 = Database('localhost', 'root', '', 'ds2copy')

    data1 = db1.query('SELECT * from categories')
    data2 = db2.query('SELECT * from categories')

    data = mergeData([data1, data2])
    print len(data1)
    print len(data)

    sql = """CREATE TABLE IF NOT EXISTS categories (
         CATEGORY  int,
         CATEGORYNAME  varchar(50))"""
    db3.query(sql)


    for row in data:
        sql = """ INSERT INTO categories (CATEGORY, CATEGORYNAME) VALUES ('%s', '%s') """
        db3.insert(sql % (row[0], row[1]))
 

if __name__ == '__main__':
    main()

