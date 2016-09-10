"""
Project: Data Science 
Subject: SQL/Python - 1. db connection

Author: Kelly Chan
Date: May 8 2014
"""

import MySQLdb

# connect database
conn = MySQLdb.connect(host='localhost', user='root', passwd='', db='ds2')
cursor = conn.cursor()  # opening cursor

n = cursor.execute('select * from cust_hist')  # querying with SQL, return # of rows
data = cursor.fetchall()  # getting all records

# connection closed
cursor.close()
conn.close()

print data
