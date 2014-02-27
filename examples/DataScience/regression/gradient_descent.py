import numpy
import pandas

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, and values for our thetas.
    """
    m = len(values)
    sum_of_square_errors = numpy.square(numpy.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """

    # Write some code here that updates the values of theta a number of times equal to
    # num_iterations.  Everytime you have computed the cost for a given set of thetas,
    # you should append it to cost_history.  The function should return both the final
    # values of theta and the cost history.

    # YOUR CODE GOES HERE
    cost_history = []
    m = len(values) * 1.0
    
    for i in range(0,num_iterations):
        # Calculate cost
        cost = compute_cost(features, values, theta)
        
        # Append cost to history
        cost_history.append(cost)
        
        # Calculate new theta
        theta = theta - (alpha/m) * numpy.dot((numpy.dot(features,theta) - values),features)

    return theta, pandas.Series(cost_history)