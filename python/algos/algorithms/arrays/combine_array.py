"""
Given 2 arrays, output an array with the elements that are contained in both

# python hasn't array data structure, but list instead
"""

def combine(array1, array2):

    array = []
    array.extend(array1)
    array.extend(array2)
    return array

if __name__ == '__main__':

    array1 = [1,2,3]
    array2 = [3,2,1]

    print combine(array1, array2)
