# -----------
# User Instructions
# 
# Modify the valid_day() function to verify 
# whether the string a user enters is a valid 
# day. The valid_day() function takes as 
# input a String, and returns either a valid 
# Int or None. If the passed in String is 
# not a valid day, return None. 
# If it is a valid day, then return 
# the day as an Int, not a String. Don't 
# worry about months of different length. 
# Assume a day is valid if it is a number 
# between 1 and 31.
# Be careful, the input can be any string 
# at all, you don't have any guarantees 
# that the user will input a sensible 
# day.
#

def valid_day(day):
    if day and day.isdigit():
        day = int(day)
        if 0 < day <= 31:
            return day



print valid_day('0') # => None    
print valid_day('1') # => 1
print valid_day('15') # => 15
print valid_day('500') # => None

