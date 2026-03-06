import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    n = len(y_true)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    diff_sqr = np.power(np.subtract(y_pred,y_true),2)
    return 1/n * np.sum(diff_sqr)
    
