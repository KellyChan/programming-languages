import sys

def reducer():
    '''
    Given the output of the mapper for this exercise, the reducer should return one row
    per unit, along with the total number of ENTRIESn_hourly over the course of may. 
    
    You can assume that the input to the reducer is sorted by UNIT, such that all rows 
    corresponding to a particular UNIT are group together.

    '''
    
    entries = 0
    old_key = None

    for line in sys.stdin:
        thisLine = line.strip().split('\t')
        
        if len(thisLine) != 2:
            continue
        
        this_key, this_entries = thisLine
        
        if old_key and old_key != this_key:
            print "{0}\t{1}".format(old_key,entries)
            entries = 0
        
        old_key = this_key
        entries += float(this_entries)
    
    if old_key != None:
        print "{0}\t{1}".format(old_key,entries)

reducer()
