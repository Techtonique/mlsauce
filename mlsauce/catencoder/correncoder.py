import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.exceptions import NotFittedError
from collections import defaultdict
try: 
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass 


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

    def __init__(
        self,
        correlation_type="spearman",
        correlation_strength=0.5,
        shrinkage=10,
        n_folds=3,
        ensemble_size=5,
        aggregate="mean",
        random_state=42,
    ):
        self.correlation_type = correlation_type
        self.correlation_strength = correlation_strength
        self.shrinkage = shrinkage
        self.n_folds = n_folds
        self.ensemble_size = ensemble_size
        self.aggregate = aggregate
        self.random_state = random_state
        self.cat_columns_ = []

        # Validate inputs
        if correlation_type not in ["spearman", "kendall"]:
            raise ValueError("correlation_type must be 'spearman' or 'kendall'")
        if not (0 <= correlation_strength <= 1):
            raise ValueError("correlation_strength must be in [0, 1]")
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1")
        if aggregate not in ["mean", "median"]:
            raise ValueError("aggregate must be 'mean' or 'median'")

    def _generate_pseudo_target(self, y, random_state):
        """Generate pseudo-target with specified rank correlation to y."""
        y = np.asarray(y)
        n = len(y)
        if n <= 1:
            return y.copy()

        # Convert to uniform margins via ranks
        ranks = rankdata(y, method="average")
        u_y = ranks / (n + 1)

        # Transform to Gaussian
        g_y = norm.ppf(u_y)

        # Convert rank correlation to Gaussian correlation
        if self.correlation_type == "spearman":
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
        z = np.quantile(y_sorted, u_z, method="linear")

        return z

    def _compute_category_statistics(self, categories, values):
        """Compute category-wise statistics with proper handling."""
        if len(categories) == 0:
            return {}

        df = pd.DataFrame({"cat": categories, "val": values})

        if self.aggregate == "mean":
            cat_stats = df.groupby("cat")["val"].agg(["mean", "count"])
            return dict(
                zip(cat_stats.index, zip(cat_stats["mean"], cat_stats["count"]))
            )
        else:  # median
            cat_stats = df.groupby("cat")["val"].agg(["median", "count"])
            return dict(
                zip(
                    cat_stats.index,
                    zip(cat_stats["median"], cat_stats["count"]),
                )
            )

    def _apply_shrinkage(self, category_stats, global_stat):
        """Apply shrinkage regularization to category statistics."""
        regularized = {}
        for cat, (stat, count) in category_stats.items():
            regularized[cat] = (count * stat + self.shrinkage * global_stat) / (
                count + self.shrinkage
            )
        return regularized

    def _identify_categorical_columns(self, X):
        """Identify categorical columns in the DataFrame."""
        cat_cols = []
        for col in X.columns:
            # Check if column is object type or has low cardinality
            if (
                X[col].dtype == "object"
                or X[col].dtype.name == "category"
                or X[col].nunique() / len(X) < 0.05
            ):  # heuristic for categorical
                cat_cols.append(col)
        return cat_cols

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

        # Identify categorical columns
        self.cat_columns_ = self._identify_categorical_columns(X)
        self.non_cat_columns_ = [
            col for col in X.columns if col not in self.cat_columns_
        ]

        # Set up cross-validation
        kf = KFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        for col in self.cat_columns_:
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
                    fold_seed = (
                        ensemble_seed + fold_idx * 1000 + hash(col) % 10000
                    )
                    z_train = self._generate_pseudo_target(
                        y_train_fold, fold_seed
                    )

                    # Compute category statistics
                    cat_stats = self._compute_category_statistics(
                        X_train_fold[col].values, z_train
                    )

                    if not cat_stats:
                        continue

                    # Apply shrinkage regularization
                    if self.aggregate == "mean":
                        global_stat = np.mean(z_train)
                    else:  # median
                        global_stat = np.median(z_train)

                    regularized_stats = self._apply_shrinkage(
                        cat_stats, global_stat
                    )

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
        if not hasattr(self, "category_mappings_"):
            raise NotFittedError(
                "This %s instance is not fitted yet." % self.__class__.__name__
            )

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

            # Only encode categorical columns, leave others unchanged
            if col in self.cat_columns_:
                mappings = self.category_mappings_[col]
                X_encoded[col] = X[col].map(mappings).fillna(self.y_mean_)
            # Non-categorical columns are left as-is

        return X_encoded

    def fit_transform(self, X, y, **fit_params):
        """Fit encoder and return encoded version of X."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not hasattr(self, "category_mappings_"):
            raise NotFittedError(
                "This %s instance is not fitted yet." % self.__class__.__name__
            )

        if input_features is None:
            return np.array(self.feature_names_in_)
        else:
            return np.array(input_features)

    def get_category_mappings(self):
        """Get the learned category mappings for inspection."""
        if not hasattr(self, "category_mappings_"):
            raise NotFittedError(
                "This %s instance is not fitted yet." % self.__class__.__name__
            )

        return self.category_mappings_.copy()

    def validate_encoding(self, X, y, plot=True):
        """
        Comprehensive validation of the encoding process, including correlation
        preservation, distribution analysis, and category-level statistics.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Input features (must be the same as used in fitting)
        y : array-like
            True target values
        plot : bool, default=True
            Whether to generate diagnostic plots
        
        Returns:
        --------
        dict
            Dictionary containing validation metrics and statistics
        """
        if not hasattr(self, "category_mappings_"):
            raise NotFittedError(
                "This %s instance is not fitted yet." % self.__class__.__name__
            )
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        
        # Generate multiple pseudo-targets for robust statistics
        pseudo_targets = []
        correlations_achieved = []
        
        for i in range(self.ensemble_size):
            seed = self.random_state + i
            z = self._generate_pseudo_target(y, seed)
            pseudo_targets.append(z)
            
            # Compute achieved correlation
            if self.correlation_type == "spearman":
                from scipy.stats import spearmanr
                corr, _ = spearmanr(y, z)
            else:  # kendall
                from scipy.stats import kendalltau
                corr, _ = kendalltau(y, z)
            correlations_achieved.append(corr)
        
        pseudo_targets = np.array(pseudo_targets)
        mean_pseudo_target = np.mean(pseudo_targets, axis=0)
        
        # Transform the data
        X_encoded = self.transform(X)
        
        # Compute overall validation metrics
        validation_results = {
            'target_correlation': self.correlation_strength,
            'achieved_correlations': correlations_achieved,
            'mean_achieved_correlation': np.mean(correlations_achieved),
            'std_achieved_correlation': np.std(correlations_achieved),
            'correlation_bias': np.mean(correlations_achieved) - self.correlation_strength,
            'original_target_stats': {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y),
                'median': np.median(y)
            },
            'pseudo_target_stats': {
                'mean': np.mean(mean_pseudo_target),
                'std': np.std(mean_pseudo_target),
                'min': np.min(mean_pseudo_target),
                'max': np.max(mean_pseudo_target),
                'median': np.median(mean_pseudo_target)
            }
        }
        
        # Category-level analysis
        category_correlations = {}
        category_stats = {}
        
        for col in self.cat_columns_:
            if col not in X.columns:
                continue
                
            unique_categories = X[col].unique()
            cat_corrs = []
            cat_means_original = []
            cat_means_pseudo = []
            cat_counts = []
            
            for category in unique_categories:
                mask = X[col] == category
                if np.sum(mask) > 5:  # Only analyze categories with sufficient samples
                    if self.correlation_type == "spearman":
                        corr, _ = spearmanr(y[mask], mean_pseudo_target[mask])
                    else:
                        corr, _ = kendalltau(y[mask], mean_pseudo_target[mask])
                    
                    cat_corrs.append(corr)
                    cat_means_original.append(np.mean(y[mask]))
                    cat_means_pseudo.append(np.mean(mean_pseudo_target[mask]))
                    cat_counts.append(np.sum(mask))
            
            category_correlations[col] = {
                'mean_correlation': np.mean(cat_corrs) if cat_corrs else np.nan,
                'std_correlation': np.std(cat_corrs) if cat_corrs else np.nan,
                'min_correlation': np.min(cat_corrs) if cat_corrs else np.nan,
                'max_correlation': np.max(cat_corrs) if cat_corrs else np.nan
            }
            
            category_stats[col] = {
                'n_categories': len(unique_categories),
                'n_analyzed_categories': len(cat_corrs),
                'category_means_original': cat_means_original,
                'category_means_pseudo': cat_means_pseudo,
                'category_counts': cat_counts
            }
        
        validation_results['category_correlations'] = category_correlations
        validation_results['category_stats'] = category_stats
        
        # Generate plots if requested
        if plot:
            try:                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.flatten()
                
                # Scatter plot: Original vs Pseudo-targets
                axes[0].scatter(y, mean_pseudo_target, alpha=0.6, s=20)
                axes[0].set_xlabel('Original Target')
                axes[0].set_ylabel('Pseudo Target')
                axes[0].set_title(f'Original vs Pseudo-targets\n{self.correlation_type.capitalize()} correlation: {validation_results["mean_achieved_correlation"]:.3f}')
                
                # Add correlation line
                z = np.polyfit(y, mean_pseudo_target, 1)
                p = np.poly1d(z)
                axes[0].plot(y, p(y), "r--", alpha=0.8)
                
                # Distribution comparison
                axes[1].hist(y, alpha=0.7, bins=30, label='Original', density=True)
                axes[1].hist(mean_pseudo_target, alpha=0.7, bins=30, label='Pseudo', density=True)
                axes[1].set_xlabel('Value')
                axes[1].set_ylabel('Density')
                axes[1].set_title('Distribution Comparison')
                axes[1].legend()
                
                # Rank comparison
                original_ranks = rankdata(y, method='average')
                pseudo_ranks = rankdata(mean_pseudo_target, method='average')
                axes[2].scatter(original_ranks, pseudo_ranks, alpha=0.6, s=20)
                axes[2].set_xlabel('Original Ranks')
                axes[2].set_ylabel('Pseudo Ranks')
                axes[2].set_title('Rank Preservation')
                
                # Category analysis - residual plot
                residuals = y - mean_pseudo_target
                # Use first categorical column for coloring if available
                if self.cat_columns_:
                    cat_col = self.cat_columns_[0]
                    unique_cats = X[cat_col].unique()[:10]  # Limit to top 10 categories
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
                    
                    for i, category in enumerate(unique_cats):
                        mask = X[cat_col] == category
                        if np.sum(mask) > 0:
                            axes[3].scatter(mean_pseudo_target[mask], residuals[mask], 
                                          alpha=0.6, s=20, color=colors[i], label=str(category))
                    
                    axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.8)
                    axes[3].set_xlabel('Pseudo Target')
                    axes[3].set_ylabel('Residuals (Original - Pseudo)')
                    axes[3].set_title('Residuals by Category')
                    axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[3].scatter(mean_pseudo_target, residuals, alpha=0.6, s=20)
                    axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.8)
                    axes[3].set_xlabel('Pseudo Target')
                    axes[3].set_ylabel('Residuals (Original - Pseudo)')
                    axes[3].set_title('Residual Plot')
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("Matplotlib/seaborn not available for plotting")
        
        return validation_results

    def get_validation_report(self, validation_results):
        """
        Generate a human-readable validation report from validation results.
        
        Parameters:
        -----------
        validation_results : dict
            Results from validate_encoding method
        
        Returns:
        --------
        str
            Formatted validation report
        """
        report = []
        report.append("=" * 60)
        report.append("RANK TARGET ENCODER VALIDATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nCORRELATION VALIDATION:")
        report.append(f"Target {self.correlation_type} correlation: {validation_results['target_correlation']:.3f}")
        report.append(f"Achieved mean correlation: {validation_results['mean_achieved_correlation']:.3f}")
        report.append(f"Correlation bias: {validation_results['correlation_bias']:.3f}")
        report.append(f"Correlation std across ensemble: {validation_results['std_achieved_correlation']:.3f}")
        
        report.append(f"\nDISTRIBUTION COMPARISON:")
        orig = validation_results['original_target_stats']
        pseudo = validation_results['pseudo_target_stats']
        report.append(f"Original target - Mean: {orig['mean']:.3f}, Std: {orig['std']:.3f}")
        report.append(f"Pseudo target  - Mean: {pseudo['mean']:.3f}, Std: {pseudo['std']:.3f}")
        
        report.append(f"\nCATEGORY-LEVEL ANALYSIS:")
        for col, stats in validation_results['category_correlations'].items():
            if not np.isnan(stats['mean_correlation']):
                report.append(f"  {col}: {stats['mean_correlation']:.3f} ± {stats['std_correlation']:.3f} "
                            f"(min: {stats['min_correlation']:.3f}, max: {stats['max_correlation']:.3f})")
        
        report.append(f"\nROBUST STATISTICS:")
        report.append(f"Ensemble size: {self.ensemble_size}")
        report.append(f"Individual correlations: {[f'{c:.3f}' for c in validation_results['achieved_correlations']]}")
        
        report.append("=" * 60)
        return "\n".join(report)