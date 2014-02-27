import numpy as np
import pandas

def normalize_features(array):
   """
   Normalize the features in our data set.
   """
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, and the values for our thetas.
    
    This should be the same code as the compute_cost function in the lesson #3 exercises. But
    feel free to implement your own.
    """
    
    # your code here
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    
    This is the same gradient descent code as in the lesson #3 exercises. But feel free
    to implement your own.
    """
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        # your code here
        # Calculate cost
        cost = compute_cost(features, values, theta)
        
        # Append cost to history
        cost_history.append(cost)
        
        # Calculate new theta
        theta = theta + (alpha / m) * np.dot((values - np.dot(features,theta)),features)
    return theta, pandas.Series(cost_history)

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, lets predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can look at information contained in the turnstile weather dataframe 
    at the link below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of .40 or better.
    
    Note: due to the memory and CPU limitation of our amazon EC2 instance, we will
    give you a random subet (~15%) of the data contained in turnstile_data_master_with_weather.csv
    
    If you receive a "server has encountered an error" message, that means you are hitting 
    the 30 second  limit that's placed on running your program. Try using a smaller number
    for num_iterations if that's the case.
    
    Or if you are using your own algorithm/modesl, see if you can optimize your code so it
    runs faster.
    '''

    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']].join(dummy_units)
    values = dataframe[['ENTRIESn_hourly']]
    m = len(values)

    features, mu, sigma = normalize_features(features)

    features['ones'] = np.ones(m)
    features_array = np.array(features)
    values_array = np.array(values).flatten()

    #Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to play with this value
    num_iterations = 75 # please feel free to play with this value

    #Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, values_array, theta_gradient_descent,
                                                            alpha, num_iterations)
    #print cost_history
    prediction = np.dot(features_array, theta_gradient_descent)

    return prediction