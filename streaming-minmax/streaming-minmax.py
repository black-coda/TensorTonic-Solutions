import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    return {
        "min" : np.full(D, np.inf),
        "max" : np.full(D, -np.inf)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch)
    X_batch_min = np.min(X_batch, axis=0)
    X_batch_max = np.max(X_batch, axis=0)


    state['min'] = np.minimum(X_batch_min, state['min'])
    state['max'] = np.maximum(X_batch_max, state['max'])

    print(f"SHAPE OF STATE MIN: {state['min'].shape}, SHAPE OF STATE MAX: {state['max'].shape}, SHAPE OF X_BATCH: {X_batch.shape}")

    normalization = (X_batch - state['min']) / ((state['max'] - state['min']) + eps)
    return normalization
    
   
