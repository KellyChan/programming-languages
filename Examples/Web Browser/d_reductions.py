# Writing Reductions

# We are looking at chart[i] and we see x => ab . cd from j.

# Hint: Reductions are tricky, so as a hint, remember that you only want to do
# reductions if cd == []

# Hint: You'll have to look back previously in the chart. 


def reductions(chart, i, x, ab, cd, j):
    # Insert code here!
    ret = []
    for jstate in chart[j]:
        next_x, next_ab, next_cd, next_j = jstate
        if cd == [] and next_cd <> [] and next_cd[0] == x:
            ret = ret + [(next_x, next_ab+[x], next_cd[1:], next_j)]
    return ret
     
    
chart = {0: [('exp', ['exp'], ['+', 'exp'], 0), ('exp', [], ['num'], 0), ('exp', [], ['(', 'exp', ')'], 0), ('exp', [], ['exp', '-', 'exp'], 0), ('exp', [], ['exp', '+', 'exp'], 0)], 1: [('exp', ['exp', '+'], ['exp'], 0)], 2: [('exp', ['exp', '+', 'exp'], [], 0)]}

print reductions(chart,2,'exp',['exp','+','exp'],[],0) == [('exp', ['exp'], ['-', 'exp'], 0), ('exp', ['exp'], ['+', 'exp'], 0)]
