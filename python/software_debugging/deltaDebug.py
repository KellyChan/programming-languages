#
# Finish the delta debug function ddmin
#


import re

def test(s):
    print s, len(s)
    if re.search("<SELECT[^>]*>", s) >= 0:
        return "FAIL"
    else:
        return "PASS"


def ddmin(s):
    assert test(s) == "FAIL"

    n = 2     # Initial granularity
    while len(s) >= 2:
        start = 0
        subset_length = len(s) / n
        some_complement_is_failing = False

        while start < len(s):
            complement = s[:start] + s[start + subset_length:]

            if test(complement) == "FAIL":
                s = complement
                n = max(n - 1, 2)
                some_complement_is_failing = True
                break
                
            start += subset_length

        if not some_complement_is_failing:
            # YOUR CODE HERE
            n = min(n * 2, len(s));
            if n == len(s):
                break
            

    return s

# UNCOMMENT TO TEST
#html_input = '<SELECT>foo</SELECT>'
#print ddmin(html_input)
