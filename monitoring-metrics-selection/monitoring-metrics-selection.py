from typing import List
import numpy as np

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    if system_type == "classification":
        return classification(y_true=y_true, y_pred=y_pred)
    elif system_type == "regression":
        return regression(y_true=y_true, y_pred=y_pred)
    elif system_type == "ranking":
        return ranking(y_pred=y_pred, y_true=y_true)
    else:
        raise ValueError()


def classification(
    y_true: List[int],
    y_pred: List[int]
):
    assert len(y_true) == len(y_pred)
    N = len(y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert np.isin(y_true, [0,1]).all()
    assert np.isin(y_pred, [0,1]).all()

    TP = np.sum((y_true == 1) & (y_pred ==1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 1) & (y_pred ==0))
    FN = np.sum((y_true == 0) & (y_pred ==1))

    accuracy = (TP + TN)/N
    precision = (TP)/(TP + FP)
    recall = TP /(TP + FN)
    F1 = (2 * precision * recall)/(precision + recall)

    return [("accuracy", accuracy.item()), ("f1", F1.item()), ("precision", precision.item()), ("recall", recall.item())]


def regression(
    y_true: List[float],
    y_pred: List[float]
):
    assert len(y_true) == len(y_pred)
    # [("mae", 0.625), ("rmse", 0.6614)]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)
    mae =   np.sum(np.abs(y_true - y_pred)) * 1 / N
    rmse = np.sqrt((np.sum(np.square(y_true - y_pred))) * 1/N)
    return [("mae", mae.item()), ("rmse", rmse.item())]


def ranking(
    y_true: List[int],
    y_pred: List[float],
    k: int = 3
):
    assert len(y_true) == len(y_pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert np.isin(y_true, [0, 1]).all()
    assert 0 < k <= len(y_true)

    # Rank by predicted score, highest first
    ranked_indices = np.argsort(y_pred)[::-1]

    # Reorder relevance according to the ranking
    ranked_relevance = y_true[ranked_indices]

    # Top-k items
    top_k = ranked_relevance[:k]

    # Number of relevant items in top-k
    relevant_at_k = np.sum(top_k)

    # Total number of relevant items
    total_relevant = np.sum(y_true)

    # Handle division by zero
    precision_at_k = (
        relevant_at_k / k
        if k > 0
        else 0.0
    )

    recall_at_k = (
        relevant_at_k / total_relevant
        if total_relevant > 0
        else 0.0
    )

    metrics = [
        (f"precision_at_{k}", precision_at_k),
        (f"recall_at_{k}", recall_at_k)
    ]

    print(f"Metrics: {metrics}")

    return sorted(metrics)