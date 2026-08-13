import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    N = len(y_true)
    cross_entropy = -np.sum(np.log(y_pred[np.arange(N), y_true])) / N
    print(cross_entropy)
    return cross_entropy