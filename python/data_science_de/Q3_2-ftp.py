"""
Project: Data Science 
Subject: Bash Scripting - 2. ftp

Author: Kelly Chan
Date: May 8 2014
"""

import os
import shutil

import MySQLdb

from ftplib import FTP
from datetime import date, timedelta


class Database:
    """ creating the class Database """

    def __init__(self, host, user, passwd, db):
        """ initializing database and connecting """
        
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

        self.conn = MySQLdb.connect(host=self.host, user=self.user, \
                                    passwd=self.passwd, db=self.db, \
                                    local_infile = 1)  # loading local file allowed
        self.cursor = self.conn.cursor()

    def insert(self, sql):
        """ inserting records into database """
        
        try:
            self.cursor.execute(sql)
            self.conn.commit()
        except:
            self.conn.rollback()

    def query(self, sql):
        """ querying data with sql """
        
        cursor = self.conn.cursor()
        cursor.execute(sql)

        return cursor.fetchall()

    def close(self):
        """ closing cursor and connection """
        
        self.cursor.close()
        self.conn.close()


def createFileName(countryList):

    today = date.today()
    fileDate = today - timedelta(days=28)

    fileNames = []
    for country in countryList:
        fileName = 'webtrekk_Marketing Report_%s_%s.zip' % (str(fileDate), str(country))
        fileNames.append(fileName)
    
    return fileNames


def ftpDownload(host, port, user, passwd, filedir, filename):
    
    ftp = FTP()
    #ftp.set_debuglevel(2)  # 2: display, 0: not display
    ftp.connect(host)  # host, port
    ftp.login()  # user, passwd

    ftp.cwd('debian')  # connecting directory
    #ftp.retrlines('LIST') # listing files under directory

    filePath = filedir + filename
    file_handler = open(filePath, 'wb').write

    ftp.retrbinary("RETR %s" % filename, file_handler)
    print "File %s has been saved in %s." % (filename, filePath)

    ftp.quit()

def copyCSV(csvfile, csvdir, todir):

    if os.path.exists(todir+csvfile):
        os.remove(todir+csvfile)
        shutil.copy(csvdir+csvfile, todir)
    else:
        shutil.copy(csvdir+csvfile, todir)
    
    print "File %s has been copied to the directory %s." % (str(csvfile), str(todir))



def main():

    countryList = ['SG', 'MY', 'TH', 'PH', 'VN']
    fileNames = createFileName(countryList)
    
    ftpDownload('ftp.debian.org', '', '', '', \
                'G:/vimFiles/python/DSInterview/DSInterview/outputs/', 'README.html')

    pyPath = 'G:/vimFiles/python/DSInterview/DSInterview/outputs/'
    execfile(pyPath + "test.py")

    csvfile = 'README.html'
    csvdir = 'G:/vimFiles/python/DSInterview/DSInterview/outputs/'
    todir = 'G:/vimFiles/python/DSInterview/DSInterview/outputs/newdir/'
    copyCSV(csvfile, csvdir, todir)

    db = Database('localhost', 'root', '', 'ds2')
    for line in open('UpdateWebtrekk.sql', 'rb').readlines():
        db.insert(line)
    db.close()


    


if __name__ == '__main__':
    main()
