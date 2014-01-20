# 
# Use `fuzzer`, `ddmin` and the given `mystery_test`
# to find the minimal length string that causes
# `mystery_test` to return 'FAIL' 
# 
# Once you've found the string, put it in the `answer`
# variable and submit.
#

answer = "x)I" # put in your answer here


from unit3 import mystery_test
import random

def fuzzer():
    string_length = int(random.random() * 1024)   # Strings up to 1024 characters long
    out = ""
    for i in range(0, string_length):
        out += chr(int(random.random() * 96 + 32)) # filled with ASCII 32..128
    return out

def ddmin(s):
    n = 2     # Initial granularity
    while len(s) >= 2:
        start = 0
        subset_length = len(s) / n
        some_complement_is_failing = False

        while start < len(s):
            complement = s[:start] + s[start + subset_length:]

            if mystery_test(complement) == "FAIL":
                s = complement
                n = max(n - 1, 2)
                some_complement_is_failing = True
                break
                
            start += subset_length

        if not some_complement_is_failing:
            if n == len(s):
                break
            n = min(n * 2, len(s))
    return s


for i in range(10000):
    f = fuzzer()
    if mystery_test(f) != "PASS":
        print ddmin(f)
