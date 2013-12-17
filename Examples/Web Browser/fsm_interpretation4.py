# Cumulative Recap!

# Suppose we want to recognize phone numbers with or without hyphens. The
# regular expression you give should work for any number of groups of any (non-
# empty) size, separated by 1 hyphen. Each group is [0-9]+.

# Hint: Accept "5" but not "-6"

import re

regexp = r"(?:[0-9]+-)*[0-9]+]*"

# regexp matches:

print re.findall(regexp,"123-4567") == ["123-4567"]
#>>> True

print re.findall(regexp,"1234567") == ["1234567"]
#>>> True

print re.findall(regexp,"08-78-88-88-88") == ["08-78-88-88-88"]
#>>> True

print re.findall(regexp,"0878888888") == ["0878888888"]
#>>> True

# regexp does not match:

print re.findall(regexp,"-6") != ["-6"]
#>>> True


