# ----------------
# User Instructions
#
# Add the appropriate return statement to the nextto(h1, h2)
# function below. It should return True when two houses 
# differ by 1, otherwise it should return False. 

import itertools

houses = [1, 2, 3, 4, 5]
orderings = list(itertools.permutations(houses))

def imright(h1, h2):
    "House h1 is immediately right of h2 if h1-h2 == 1."
    return h1-h2 == 1

def nextto(h1, h2):
    "Two houses are next to each other if they differ by 1."
    return abs(h1-h2) == 1
