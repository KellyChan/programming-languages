"""
Find the kth largest element in an array.

Given an array and a number k where k is smaller than size of array, 
we need to find the kth smallest element in the given array. 
It is given that ll array elements are distinct.

Input: arr[] = {7, 10, 4, 3, 20, 15}
       k = 3
Output: 7

Input: arr[] = {7, 10, 4, 3, 20, 15}
       k = 4
Output: 10

#-------------------------------------------------------------------#


Find the k largest element in an array.

For example, if given array is [1, 23, 12, 9, 30, 2, 50] and you are asked for the largest 3 elements i.e., k = 3 then your program should print 50, 30 and 23.
"""


def min_k(num_list, k):
    "Time complexity: O(N Log N)"
    return sorted(num_list)[k-1]

def mins_k(num_list, k):
    return sorted(num_list)[:k]

def max_k(num_list, k):
    return sorted(num_list)[-k]

def maxs_k(num_list, k):
    return sorted(num_list)[-k:]


if __name__ == '__main__':

    num_list = [8,3,43,2]

    print min_k(num_list, 2)
    print mins_k(num_list, 2)
    print max_k(num_list, 2)
    print maxs_k(num_list, 2)
