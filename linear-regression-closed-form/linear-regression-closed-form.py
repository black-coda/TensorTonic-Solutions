import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.asarray(X)
    y=np.asarray(y)
    bracket = X.T @ X
    return np.linalg.inv(bracket).dot(X.T.dot(y))
    