import numpy as np
from typing import List

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x)
    return 1/(1+np.exp(-x))