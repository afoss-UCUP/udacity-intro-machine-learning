#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    # get squared residuals    
    error = (net_worths - predictions)
    
    # order the index of squared residuals (descending)
    ordered_ind = sorted(range(len(error**2)), key=lambda k: (error**2)[k], reverse = True)
    
    # find the point to begin keeping observations 
    ind_beg = int(.1 * len(ages))
    
    cleaned_data = np.concatenate((ages, net_worths, error), axis = 1)[ordered_ind[(ind_beg):],:]
    
    ### your code goes here
    
    return cleaned_data

