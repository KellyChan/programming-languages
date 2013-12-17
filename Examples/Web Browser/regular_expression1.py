# Disjunction Construction

import re

# Assign to the variable regexp a regular expression that matches either the
# exact string ab or one or more digits.

regexp = r"[a][b]|[0-9]+"

# regexp matches:

print re.findall(regexp,"ab") == ["ab"]
#>>> True

print re.findall(regexp,"1") == ["1"]
#>>> True

print re.findall(regexp,"123") == ["123"]
#>>> True


# regexp does not match:

print re.findall(regexp,"a") != ["a"]
#>>> True

print re.findall(regexp,"abc") != ["abc"]
#>>> True

print re.findall(regexp,"abc123") != ["abc123"]
#>>> True 
