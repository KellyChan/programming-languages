# the variance function to make it return the variance of a list of numbers
 
def mean(data):
    return sum(data)/len(data)
 
def variance1(data):
    sqdist = 0
    for i in range(len(data)):
        distance = data[i] - mean(data)
        sqdist = sqdist + distance * distance
    return sqdist/len(data)
 
def variance2(data)
    mu = mean(data)
    ndata = []
    for i in range(len(data)):
        ndata.append((data[i] - mu)**2)
    sigma2 = mean(ndata)
    return sigma2
    
def variance3(data)
    mu = mean(data)
    return mean([(x - mu)**2 for x in data])
    