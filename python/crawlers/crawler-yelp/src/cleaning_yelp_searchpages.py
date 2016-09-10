"""
Author: Kelly Chan
Date: July 14 2014

Project: Cleansing the data from the extracted attributes (by Pages)

- url: http://www.yelp.com/search?find_loc=los+angeles&cflt=restaurants&start=%s
- %s: the number of the pages

steps:
- 1. getting the data extracted from the html contents
- 2. cleansing the data by attributes
- 3. saving the cleaned data (.tsv) to the local drive

"""

def getData(datafile):
    """ reading the .txt data and then return it """

    data = []
    with open(datafile, 'rb') as f:
        for line in f.readlines():
            thisLine = line.strip().split('],[')
            data.append(thisLine)
    return data

def cleanLeftBracket(attribute):
    """ cleansing the left bracket """
    
    return attribute.replace("['", "").replace("'", "")

def cleanRightBracket(attribute):
    """ cleansing the right bracket """
    
    return attribute.replace("]", "")

def cleanString(attribute):
    """ cleansing the string """
    
    return attribute.replace("'", "")

def cleanAttributes(data):
    """ cleansing the attributes one by one and then return the cleaned data """

    cleanedData = []

    for record in data[1:]:

        # index
        index = cleanLeftBracket(record[0]).strip()
        #print index

        # link
        link = cleanString(record[1]).strip()
        link = "http://www.yelp.com/" + link
        #print link
        
        # name
        name = cleanString(record[2]).strip()
        #print name

        # rating
        rating = cleanString(record[3]).strip()
        #print rating

        # review
        review = cleanString(record[4]).strip()
        #print review

        # price
        price = cleanString(record[5]).strip()
        #print price

        # category
        categories = cleanString(record[6]).strip()
        #print categories

        # neighborhood
        neighborhood = cleanString(record[7]).strip()
        #print neighborhood

        # address
        address = cleanString(record[8]).strip()
        address = address.replace("\\n        </br>", "")
        address = address.replace("\\n", "")
        address = address.split('<br>')
        if len(address) == 1:
            road = ""
            mainAddress = address[0]
        elif len(address) == 2:
            road = address[0].strip()
            mainAddress = address[1]
        #print road
 
        mainAddress = mainAddress.split(",")
        city = mainAddress[0].strip()
        state, postCode = mainAddress[1].strip().split(" ")
        #print city
        #print state, postCode

        # phone
        phone = cleanString(record[9]).strip()
        #print phone

        # feedback
        feedback = cleanRightBracket(record[10]) 
        #print feedback

        cleanedData.append([index, link, name, rating, review, price, categories, \
                     neighborhood, road, city, postCode, phone, feedback])

    return cleanedData

def outTSV(data, outPath, fileName):
    """ output the data as .tsv file """

    header = "index\tlink\tname\trating\treview\tprice\tcategories\tneighborhood\troad\tcity\tpostcode\tphone\tfeedback\n"

    with open(outPath+fileName, "wb") as f:
        f.write(header)
        for record in data:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % \
                    (record[0],record[1],record[2],record[3],record[4],record[5],record[6],\
                     record[7],record[8],record[9],record[10],record[11],record[12]))


def main():

    dataPath = "path/regexEval/data/"
    fileName = "raw_pages_data.txt"
    outName = "cleaned_pages_data.tsv"

    data = getData(dataPath+fileName)
    cleanedData = cleanAttributes(data)
    outTSV(cleanedData, dataPath, outName)
 

if __name__ == '__main__':
    main()