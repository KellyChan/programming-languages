import MySQLdb

class Database:
    """ creating the class Database """

    def __init__(self, host, user, passwd, db):
        """ initializing database and connecting """
        
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

        self.conn = MySQLdb.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db)
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


def mergeData(dataList):
    """ merging all country data into one """
    
    fullData = []
    for data in dataList:
        fullData.extend(data)
    return fullData


def main():
    
    # creating the instances of database
    dbMY = Database('localhost', 'root', 'pw_root', 'zlmy_reporting')
    dbSG = Database('localhost', 'root', 'pw_root', 'zlsg_reporting')
    dbRE = Database('localhost', 'root', 'pw_root', 'zlre_reporting')

    # querying data from database
    dataMY = db1.query('SELECT * from orders')
    dataSG = db2.query('SELECT * from orders')

    # merging data into one
    data = mergeData([dataMY, dataSG])

    # creating new table if not exists in the database zlre_reporting
    sql = """
             CREATE TABLE IF NOT EXISTS orders (
               Days  VARCHAR(10),
               Order_nr  INT(20),
               Price_paid  FLOAT
             )
         """
    dbRE.query(sql)

    # inserting records into the database zlre_reporting
    sql = """
             INSERT INTO orders (Days, Order_nr, Price_paid) VALUES ('%s', '%s', '%s')
          """
    for row in data:
        db3.insert(sql % (row[0], row[1], row[2]))
    
    
    # closing database
    dbMY.close()
    dbSG.close()
    dbRE.close()
 

if __name__ == '__main__':
    main()