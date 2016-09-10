import numpy as np

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    # YOUR CODE GOES HERE
    # R^2 = 1 - sum((yi - f)^2) / sum((yi - ymean)**2)
    
    s1 = np.sum((data - predictions)**2)
    s2 = np.sum((data - np.mean(data))**2)
    r_squared = 1 - s1/s2

    return r_squared