import numpy as np

def softmax(x):
    """
    Compute the softmax of x x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    if x.ndim == 1:
        max = np.max(x)
        return np.exp(x - max)/(np.sum(np.exp(x-max)))
    else:
        return np.exp(x - np.max(x, axis=1).reshape(-1, 1)) / np.sum(np.exp(x-np.max(x,axis=1).reshape(-1,1)), axis=1).reshape(-1, 1)
        