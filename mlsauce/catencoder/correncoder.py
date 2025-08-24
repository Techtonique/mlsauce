import numpy as np
from scipy.stats import norm, rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.exceptions import NotFittedError
from collections import defaultdict


class RankTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Rank-based target encoder using Spearman rho or Kendall tau via 
    Gaussian copula with proper cross-validation.
    
    This encoder uses cross-validation and pseudo-targets generated via 
    Gaussian copula with specified rank correlation to create robust, 
    regularized encodings that prevent overfitting.
    
    Parameters:
    -----------
    correlation_type : str, default='spearman'
        Type of rank correlation ('spearman' or 'kendall').
    correlation_strength : float, default=0.5
        Desired strength of rank correlation (between 0 and 1).
    shrinkage : float, default=10
        Shrinkage parameter for regularization (Bayesian average).
    n_folds : int, default=3
        Number of CV folds for leakage-free encoding.
    ensemble_size : int, default=5
        Number of pseudo-targets to average over (reduces variance).
    aggregate : str, default='mean'
        Aggregation method for combining values within categories ('mean' or 'median').
    random_state : int, default=42
        Random seed for reproducibility.
    """    
    def __init__(self, correlation_type='spearman', 
                 correlation_strength=0.5,
                 shrinkage=10, n_folds=3, ensemble_size=5, 
                 aggregate='mean',
                 random_state=42):
        self.correlation_type = correlation_type
        self.correlation_strength = correlation_strength
        self.shrinkage = shrinkage
        self.n_folds = n_folds
        self.ensemble_size = ensemble_size
        self.aggregate = aggregate
        self.random_state = random_state
        
        # Validate inputs
        if correlation_type not in ['spearman', 'kendall']:
            raise ValueError("correlation_type must be 'spearman' or 'kendall'")
        if not (0 <= correlation_strength <= 1):
            raise ValueError("correlation_strength must be in [0, 1]")
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1")
        if aggregate not in ['mean', 'median']:
            raise ValueError("aggregate must be 'mean' or 'median'")

    def _generate_pseudo_target(self, y, random_state):
        """Generate pseudo-target with specified rank correlation to y."""
        y = np.asarray(y)
        n = len(y)
        if n <= 1:
            return y.copy()

        # Convert to uniform margins via ranks
        ranks = rankdata(y, method='average')
        u_y = ranks / (n + 1)
        
        # Transform to Gaussian
        g_y = norm.ppf(u_y)

        # Convert rank correlation to Gaussian correlation
        if self.correlation_type == 'spearman':
            rho_g = 2 * np.sin(np.pi * self.correlation_strength / 6)
        else:  # kendall
            rho_g = np.sin(np.pi * self.correlation_strength / 2)
        rho_g = np.clip(rho_g, -1.0, 1.0)

        # Generate correlated Gaussian variable
        rng = np.random.RandomState(random_state)
        eta = rng.normal(size=n)
        g_z = rho_g * g_y + np.sqrt(1 - rho_g**2) * eta
        
        # Transform back to original scale via quantiles
        u_z = norm.cdf(g_z)
        y_sorted = np.sort(y)
        z = np.quantile(y_sorted, u_z, method='linear')
        
        return z

    def _compute_category_statistics(self, categories, values):
        """Compute category-wise statistics with proper handling."""
        if len(categories) == 0:
            return {}
        
        df = pd.DataFrame({'cat': categories, 'val': values})
        
        if self.aggregate == 'mean':
            cat_stats = df.groupby('cat')['val'].agg(['mean', 'count'])
            return dict(zip(cat_stats.index, 
                           zip(cat_stats['mean'], cat_stats['count'])))
        else:  # median
            cat_stats = df.groupby('cat')['val'].agg(['median', 'count'])
            return dict(zip(cat_stats.index, 
                           zip(cat_stats['median'], cat_stats['count'])))

    def _apply_shrinkage(self, category_stats, global_stat):
        """Apply shrinkage regularization to category statistics."""
        regularized = {}
        for cat, (stat, count) in category_stats.items():
            regularized[cat] = (count * stat + self.shrinkage * global_stat) / (count + self.shrinkage)
        return regularized

    def fit(self, X, y):
        """Fit the encoder using cross-validation to prevent leakage."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        X = X.reset_index(drop=True)  # Ensure clean integer indices
        y = np.asarray(y)
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")

        self.feature_names_in_ = list(X.columns)
        self.y_mean_ = np.mean(y) if len(y) > 0 else 0.0
        self.category_mappings_ = {}

        # Set up cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for col in X.columns:
            if X[col].nunique() <= 1:
                # Handle constant columns
                self.category_mappings_[col] = {X[col].iloc[0]: self.y_mean_}
                continue
            
            # Collect encodings for each category across all CV folds and ensemble members
            category_encodings = defaultdict(list)
            
            for ensemble_idx in range(self.ensemble_size):
                ensemble_seed = self.random_state + ensemble_idx
                fold_encodings = np.full(len(y), np.nan)
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                    # Training data for this fold
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    
                    if len(y_train_fold) == 0:
                        continue
                    
                    # Generate pseudo-target for this fold and ensemble member
                    # Use deterministic seed based on ensemble_idx, fold_idx, and column
                    fold_seed = ensemble_seed + fold_idx * 1000 + hash(col) % 10000
                    z_train = self._generate_pseudo_target(y_train_fold, fold_seed)
                    
                    # Compute category statistics
                    cat_stats = self._compute_category_statistics(
                        X_train_fold[col].values, z_train)
                    
                    if not cat_stats:
                        continue
                    
                    # Apply shrinkage regularization
                    if self.aggregate == 'mean':
                        global_stat = np.mean(z_train)
                    else:  # median
                        global_stat = np.median(z_train)
                    
                    regularized_stats = self._apply_shrinkage(cat_stats, global_stat)
                    
                    # Encode validation fold
                    for idx in val_idx:
                        category = X_val_fold.loc[idx, col]
                        if category in regularized_stats:
                            fold_encodings[idx] = regularized_stats[category]
                        else:
                            fold_encodings[idx] = global_stat
                
                # Collect encodings by category for this ensemble member
                for idx, encoding in enumerate(fold_encodings):
                    if not np.isnan(encoding):
                        category = X.iloc[idx][col]
                        category_encodings[category].append(encoding)
            
            # Average encodings for each category across all ensemble members and folds
            final_mappings = {}
            for category, encodings in category_encodings.items():
                if encodings:
                    final_mappings[category] = np.mean(encodings)
                else:
                    final_mappings[category] = self.y_mean_
            
            self.category_mappings_[col] = final_mappings

        return self

    def transform(self, X):
        """Transform categorical columns using learned encodings."""
        if not hasattr(self, 'category_mappings_'):
            raise NotFittedError("This %s instance is not fitted yet." % self.__class__.__name__)

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # Check for missing columns
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns from training: {missing_cols}")
        
        X_encoded = X.copy()

        for col in self.feature_names_in_:
            if col not in X.columns:
                # This shouldn't happen due to check above, but be safe
                X_encoded[col] = self.y_mean_
                continue
                
            # Map categories to encodings, unseen categories get global mean
            mappings = self.category_mappings_[col]
            X_encoded[col] = X[col].map(mappings).fillna(self.y_mean_)

        return X_encoded

    def fit_transform(self, X, y, **fit_params):
        """Fit encoder and return encoded version of X."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not hasattr(self, 'category_mappings_'):
            raise NotFittedError("This %s instance is not fitted yet." % self.__class__.__name__)
        
        if input_features is None:
            return np.array(self.feature_names_in_)
        else:
            return np.array(input_features)
    
    def get_category_mappings(self):
        """Get the learned category mappings for inspection."""
        if not hasattr(self, 'category_mappings_'):
            raise NotFittedError("This %s instance is not fitted yet." % self.__class__.__name__)
        
        return self.category_mappings_.copy()