import sys
import string

def mapper():
    """
    The input to this mapper will be the final Subway-MTA dataset.  This mapper 
    should return a line for each UNIT, along with the number of ENTRIESn_hourly for that row.
    
    An example input to the mapper may would look like this:
    R002    1050105.0
    
    The mapper should emit a key and value pair separated by a tab, for example:
    R002\t105105.0
    """
    
    for line in sys.stdin:
        # your code here
        thisLine = line.strip().split(',')
        if len(thisLine) != 22 or thisLine[6] == 'ENTRIESn_hourly':
            continue
        
        print "{0}\t{1}".format(thisLine[1], thisLine[6])
                                    

mapper()