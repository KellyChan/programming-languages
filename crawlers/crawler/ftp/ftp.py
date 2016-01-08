import os
import shutil
from ftplib import FTP
from datetime import date, timedelta

import MySQLdb

class Database:
    """ creating the class Database """

    def __init__(self, host, user, passwd, db):
        """ initializing database and connecting """
        
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

        self.conn = MySQLdb.connect(host=self.host, 
                                    user=self.user, 
                                    passwd=self.passwd, 
                                    db=self.db, 
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
    """ generating the names of zip files """

    today = date.today()
    fileDate = today - timedelta(days=28)  # rolling 28 days ago

    fileNames = []
    for country in countryList:
        fileName = 'webtrekk_Marketing Report_%s_%s.zip' % (str(fileDate), str(country))
        fileNames.append(fileName)
    
    return fileNames


def ftpDownload(host, port, user, passwd, remotedir, localdir, filename):
    """ downloading zip files from ftp and then saving them to the directory """
    
    ftp = FTP()
    #ftp.set_debuglevel(2)  # 2: display, 0: not display
    ftp.connect(host, port)  # host, port
    ftp.login(user, passwd)  # user, passwd

    ftp.cwd(remotedir)  # connecting the directory of files
    #ftp.retrlines('LIST') # listing files under directory

    # creating local file path
    localPath = localdir + filename
    file_handler = open(localPath, 'wb').write

    # downloading files from ftp and saving to local path
    ftp.retrbinary("RETR %s" % filename, file_handler)
    print "File %s has been saved in %s." % (filename, localPath)

    ftp.quit()

    
def copyCSV(csvfile, csvdir, todir):
    """ copying csv file to mysql directory"""

    if os.path.exists(todir+csvfile):
        # if csv file exists, remove csv file and copy the new one
        os.remove(todir+csvfile)
        shutil.copy(csvdir+csvfile, todir)
    else:
        # if csv file does not exist, copy csv file to the directory
        shutil.copy(csvdir+csvfile, todir)
    
    print "File %s has been copied to the directory %s." % (str(csvfile), str(todir))
    
    
def main():

    # step1. creating file names
    countryList = ['SG', 'MY', 'TH', 'PH', 'VN']
    fileNames = createFileName(countryList)
    
    # step2. downloading files to local directory
    for fileName in fileNames:
        ftpDownload('10.11.12.13', '22', 'hello', 'world', 'XXX/XXX', 
                    'home/Marketing Report/Data/', fileName)

    # step3. running python script
    pyPath = 'home/Marketing Report/'
    execfile(pyPath + 'ZMR.py')

    # step4. copying csv file to mysql directory
    csvfile = 'WT.csv'
    csvdir = 'home/Marketing/Report'
    todir = 'mysql_direcotry'
    copyCSV(csvfile, csvdir, todir)
    
    # step5. log onmysql with local-infile=1 and run query
    db = Database('localhost', 'root', 'pw_root', 'test')
    for line in open('UpdateWebtrekk.sql', 'rb').readlines():
        db.insert(line)
    db.close()    


if __name__ == '__main__':
    main()
