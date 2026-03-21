import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Fix 1D case
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Compute distances
    # ∣∣q−x∣∣2=∣∣q∣∣2+∣∣x∣∣2−2q⋅x
    train_sq = np.sum(X_train**2, axis=1)
    test_sq = np.sum(X_test**2, axis=1)

    distances = test_sq[:, np.newaxis] + train_sq - 2 * X_test @ X_train.T

    # Get sorted indices
    sorted_idx = np.argsort(distances, axis=1)

    n_train = X_train.shape[0]

    if k <= n_train:
        return sorted_idx[:, :k]
    else:
        # Take all available neighbors
        result = sorted_idx

        # Pad with -1
        pad_width = k - n_train
        padding = -1 * np.ones((X_test.shape[0], pad_width), dtype=int)

        return np.hstack([result, padding])