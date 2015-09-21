import sys
import string

def mapper():
    '''
    In this exercise, we want to sum up all the values in the ENTRIEsn_hourly column in the 
    turnstile_weather.csv. You can check out the csv file and its structure below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    
    Each line of input will look like a row from our final Subway-MTA dataset, in csv format.
    
    This mapper should
    1) return 0 as the key and
    2) the number in the ENTRIEsn_hourly column as the value
    3)  The key and the count should be separated by a tab
        for example: 0\t12345

    Example output to the reducer would look like this:
    0   10501050105010.0
    '''
    for line in sys.stdin:
    	# your code here
        thisLine = line.strip().split(',')
        
        if len(thisLine) != 22 or thisLine[6] == 'ENTRIESn_hourly':
            continue
        
        
        print "{0}\t{1}".format(0, thisLine[6])

mapper()
