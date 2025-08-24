from sklearn.base import clone
import numpy as np
from sklearn.model_selection import cross_val_score


def penalized_cross_val_score(
    estimator,
    X,
    y,
    param_dict,
    cv=5,
    scorer=None,
    penalty_strength=0.1,
    penalty_type="ci",
    greater_is_better=False,
):
    """
    Calculates a penalized cross-validation score that balances mean performance
    and result stability (low variability across folds).

    Parameters:
    -----------
    estimator : sklearn estimator
        Model to evaluate.
    X, y : array-like
        Training data.
    param_dict : dict
        Hyperparameters to set on the estimator.
    cv : int, default=5
        Number of cross-validation folds (must be >= 2).
    scorer : callable or str, optional
        Scikit-learn scorer (e.g., from sklearn.metrics.make_scorer).
    penalty_strength : float, default=0.1
        Multiplicative factor for the variability penalty.
        penalty_strength=0.1 = penalize by up to 10% of mean score
    penalty_type : {'std', 'max', 'range', 'ci'}
        Type of variability to penalize:
        - 'std': standard deviation of fold scores
        - 'max': maximum deviation from mean across folds
        - 'range': difference between best and worst fold
        - 'ci': approximate 95% confidence interval width (2 * SEM)
    greater_is_better : bool
        Whether higher raw scores are better (e.g., accuracy=True, RMSE=False).

    Returns:
    --------
    penalized_score : float
        Mean CV score adjusted by penalty. Always "lower is better" in effect,
        so unstable models are penalized.
    """

    if penalty_strength < 0:
        raise ValueError("penalty_strength must be non-negative.")

    if cv < 2:
        raise ValueError("cv must be at least 2.")

    # Validate parameters
    estimator_params = estimator.get_params()
    missing_params = [key for key in param_dict if key not in estimator_params]
    if missing_params:
        raise ValueError(
            f"Estimator does not have parameters: {', '.join(missing_params)}"
        )

    # Clone and configure estimator
    current_estimator = clone(estimator)
    current_estimator.set_params(**param_dict)

    # Perform cross-validation
    cv_scores = cross_val_score(current_estimator, X, y, cv=cv, scoring=scorer)

    if len(cv_scores) == 0:
        raise ValueError("Cross-validation scores are empty.")

    mean_score = np.mean(cv_scores)

    # Compute variability measure
    if penalty_type == "std":
        variability_measure = np.std(cv_scores)
    elif penalty_type == "max":
        variability_measure = np.max(np.abs(cv_scores - mean_score))
    elif penalty_type == "range":
        variability_measure = np.ptp(cv_scores)  # max - min
    elif penalty_type == "ci":
        # Approximate 95% CI width: 2 * standard error of the mean
        variability_measure = 2 * (np.std(cv_scores) / np.sqrt(len(cv_scores)))
    else:
        raise ValueError("penalty_type must be 'std', 'max', 'range', or 'ci'.")

    # Scale penalty relative to mean score magnitude
    if abs(mean_score) > 1e-10:  # avoid division by zero
        normalized_penalty = penalty_strength * (
            variability_measure / abs(mean_score)
        )
    else:
        normalized_penalty = penalty_strength * variability_measure

    # Apply penalty: make worse for instability
    if greater_is_better:
        return mean_score - normalized_penalty  # lower score = penalized
    else:
        return mean_score + normalized_penalty  # higher score = penalized
