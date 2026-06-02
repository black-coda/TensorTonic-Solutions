import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m, n = X.shape
    
    # 1. Initialize weights (w) and bias (b) to zeros
    w = np.zeros(n)
    b = 0.0
    
    # 2. Gradient Descent Loop
    for _ in range(steps):
        # Calculate the linear combination
        z = np.dot(X, w) + b
        
        # Apply the numerically stable sigmoid function
        y_pred = _sigmoid(z)
        
        # 3. Calculate gradients
        # Error vector (predictions - actual targets)
        error = y_pred - y
        
        # Partial derivatives
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
        
        # 4. Update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, b