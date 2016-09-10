"""
Squaring all elements in a list.
"""

import numpy as np

def square_loop(num_list):
    return [number ** 2 for number in num_list]

def square_map(num_list):
    return map(pow, num_list, [2]*len(num_list))

def square_lambda(num_list):
    return map(lambda x: x ** 2, num_list)


def square_yield(num_list):
    for number in num_list: 
        yield number ** 2

def square_np(num_list):
    return list(np.array(num_list)**2)

if __name__=='__main__':

    num_list = [1, 2, 3]

    print "square_loop: %s" % square_loop(num_list)
    print "square_map: %s" % square_map(num_list)
    print "square_lambda: %s" % square_lambda(num_list)
    print "square_yield: %s" % square_yield(num_list)
    print "square_np: %s" % square_np(num_list)
