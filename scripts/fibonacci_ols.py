import numpy as np

def make_xy(seq):
    """
    Creates lagged features for OLS.
    Given [1, 1, 2, 3, 5], it might create:
    X = [[1, 1], [1, 2], [2, 3]], y = [2, 3, 5]
    """
    X, y = [], []
    for i in range(len(seq) - 2):
        X.append([seq[i], seq[i+1]])
        y.append(seq[i+2])
    return np.array(X), np.array(y)

def roll_forecast(model, seq, n_preds):
    """
    Recursively predicts the next n_preds values.
    Returns the full sequence (initial + predictions).
    """
    result = list(seq)
    for _ in range(n_preds):
        X_next = np.array([[result[-2], result[-1]]])
        y_next = model.predict(X_next)[0]
        result.append(y_next)
    return result
