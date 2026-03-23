import numpy as np



def median_impute(X: np.ndarray):
    """
    Fill NaN values in each feature column using column median.
    """
    filled = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    return np.where(np.isnan(filled), 0, filled)


def mean_impute(X: np.ndarray) -> np.ndarray:
    """
    Fill NaN values in each feature column using column mean.
    """
    filled = np.where(np.isnan(X), np.mean(X, axis=0, where=~np.isnan(X)), X)
    return np.where(np.isnan(filled), 0, filled)

def impute_missing(X: np.ndarray, strategy: str = "mean"):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    if strategy == "mean":
        return mean_impute(X)
    else:
        return median_impute(X)