import sys

def reducer():
    '''
    Given the output of the mapper for this assignment, simply print
    0 and then the average number of riders per day for the month of 05/2011,
    separated by a tab.
    
    There are 31 days in 05/2011.
    
    Example output might look like this:
    0   10501050.0
    '''

    riders = 0
    old_key = None

    # your code here
    for line in sys.stdin:
        riders += float(line.split('\t')[1])
    riders = riders / 31
    print '{0}\t{1}'.format(0, riders)

reducer()
