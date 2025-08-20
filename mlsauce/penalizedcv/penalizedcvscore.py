import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier

def penalized_cross_val_score(estimator, X, y, param_dict, cv=5,
                              scorer=None, penalty_strength=0.0, penalty_type='std'):
    """
    Calculates a penalized cross-validation score.

    Parameters:
    estimator: The scikit-learn-like estimator.
    X, y: The training data.
    param_dict: A dictionary of hyperparameters to set (e.g., {'alpha': 0.1, 'C': 1.0}).
    cv: Number of cross-validation folds.
    scorer: A scikit-learn scorer object (default None, uses estimator's default scoring).
    penalty_strength: The strength of the penalty (e.g., 1.0, 0.5). 
                      Controls how much to penalize instability or high scores.
    penalty_type: Type of penalty to apply. 
                 'std'  : Penalizes the standard deviation of the CV scores.
                 'max'  : Penalizes the maximum (best) fold score.
                 'range': Penalizes the range (max - min) of the CV scores.

    Returns:
    penalized_score: The mean CV score, adjusted by the chosen penalty.
    """
    
    # Ensure that all parameters in the dictionary exist in the estimator
    missing_params = [key for key in param_dict if not hasattr(estimator, key)]
    if missing_params:
        raise ValueError(f"Estimator does not have parameters: {', '.join(missing_params)}")
    
    # Set the parameters on a clone of the estimator to avoid side effects
    current_estimator = estimator.set_params(**param_dict)
    
    # Perform cross-validation
    cv_scores = cross_val_score(current_estimator, X, y, cv=cv, scoring=scorer)
    
    # Check if cv_scores are valid
    if cv_scores.size == 0:
        raise ValueError("Cross-validation scores are empty. Please check your dataset or cross-validation setup.")
    
    mean_score = np.mean(cv_scores)
    
    # Calculate the chosen penalty term
    if penalty_type == 'std':
        penalty_term = np.std(cv_scores)
    elif penalty_type == 'max':
        penalty_term = np.max(cv_scores) # This will be a large positive number for good scores
    elif penalty_type == 'range':
        penalty_term = np.ptp(cv_scores) # Peak-to-peak (max - min)
    else:
        raise ValueError("penalty_type must be 'std', 'max', or 'range'")    
    # Apply the penalty. 
    return mean_score - (penalty_strength * penalty_term)
