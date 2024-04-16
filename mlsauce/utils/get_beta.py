import numpy as np

def get_beta(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)
