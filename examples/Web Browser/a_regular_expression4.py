# Tricky REs with ^ and \

# Assign to regexp a regular expression for double-quoted string literals that
# allows for escaped double quotes.

# Hint: Escape " and \
# Hint: (?: (?: ) )

import re

#regexp = r'".*[^\\]"'
regexp = r'"(?:[^\\]|(?:\\.))*"'

# regexp matches:

print re.findall(regexp,'"I say, \\"hello.\\""') == ['"I say, \\"hello.\\""']
#>>> True


# regexp does not match:

print re.findall(regexp,'"\\"') != ['"\\"']
#>>> True

