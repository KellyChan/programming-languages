"""
Given a list of numbers, get all possible pairs of numbers?
What is your solutions' time complexity?
"""

def all_pairs(num_list):
    """
    time complexity: O(n*n)

    >>> all_pairs([1,2,3,4,5])
    """

    list_len = len(num_list)

    for i in range(list_len):
        for j in range(i+1, list_len):
            print (num_list[i], num_list[j])

#------------------------------------------------------------#

import itertools

def all_pairs_opt1(num_list):
    print list(itertools.combinations(num_list, 2))

if __name__ == '__main__':

    num_list = [1,2,3,4,5]
    
    # all_pairs(num_list)
    all_pairs_opt1(num_list)

