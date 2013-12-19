# RE Challenges

# Assign to the variable regexp a Python regular expression that matches single-
# argument mathematical functions.

# The function name is a lowercase word (a-z), the function argument must be a
# number (0-9), and there may optionally be spaces before and/or after the
# argument.

# Hint: You may need to escape the ( and ).

import re

regexp = r"[a-z]+\( *[0-9]+ *\)"

# regexp matches:

print re.findall(regexp,"cos(0)") == ["cos(0)"]
#>>> True

print re.findall(regexp,"sqrt(   2     )") == ["sqrt(   2     )"]
#>>> True


# regexp does not match:

print re.findall(regexp,"cos     (0)") != ["cos     (0)"]
#>>> True

print re.findall(regexp,"sqrt(x)") != ["sqrt(x)"]
#>>> True


        
