# 
# Finish the simplify function.
# Have it deal with the case where test(s1) doesn't fail
# 
# If neither s1 or s2 cause the test to fail then return 
# the original string, s.

import re

def test(s):
    print `s`, len(s)
    if re.search("<SELECT[^>]*>", s) >= 0:
        return "FAIL"
    else:
        return "PASS"

def simplify(s):
    assert test(s) == "FAIL"
    
    split = len(s) / 2
    s1 = s[:split]
    s2 = s[split:]
    
    if test(s1) == "FAIL":
        return simplify(s1)
    # YOUR CODE HERE
    elif test(s2) == "FAIL":
        return simplify(s2)
    
    return s

# UNCOMMENT TO TEST
#html_input = '<SELECT>foo</SELECT>'
#print simplify(html_input)
