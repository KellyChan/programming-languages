import sys

def reducer():
    '''
    For every single unit, write a reducer that will return the busiest datetime (e.g.,
    the entry that had the most entries).  Ties should go to datetimes that are later on in the 
    Month of May.  You can assume that the contents of the reducer will be sorted by UNIT, such that
    all entries corresponding to a given UNIT will be sorted together. The output should be the UNIT
    name, the datetime (which is a concatenation of date and time), and ridership separated by tabs.

    For example, the output of the reducer may look like this:
    R001    2011-05-11 17:00:00	   31213.0
    R002	2011-05-12 21:00:00	   4295.0
    R003	2011-05-05 12:00:00	   995.0
    R004	2011-05-12 12:00:00	   2318.0
    R005	2011-05-10 12:00:00	   2705.0
    R006	2011-05-25 12:00:00	   2784.0
    R007	2011-05-10 12:00:00	   1763.0
    R008	2011-05-12 12:00:00	   1724.0
    R009	2011-05-05 12:00:00	   1230.0
    R010	2011-05-09 18:00:00	   30916.0
    ...
    ...
    
    '''

    max_entries = 0
    old_key = None
    datetime = ''

    # your code here
    for line in sys.stdin:
        thisLine = line.strip().split('\t')
        
        if len(thisLine) != 4:
            continue
        
        this_key, this_riders, this_date, this_time = thisLine
        this_datetime = this_date + ' ' + this_time
        
        if old_key and old_key != this_key:
            print "{0}\t{1}\t{2}".format(old_key, datetime, max_entries)
            max_entries = 0
            datetime = ''
        
        old_key = this_key
        this_riders = float(this_riders)
        if this_riders >= max_entries:
            max_entries = this_riders
            datetime = this_datetime
            
    if old_key != None:
        print "{0}\t{1}\t{2}".format(old_key, datetime, max_entries)
            
    
reducer()
