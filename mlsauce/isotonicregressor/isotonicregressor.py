import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class IsotonicRegressor(BaseEstimator, RegressorMixin):
    """Isotonic Regressor with postprocessing.

    This class takes a base regressor and applies isotonic regression as
    postprocessing in the predict method. The isotonic regression ensures
    that the predictions are monotonically increasing or decreasing.

    Attributes:
        regr: estimator
            Base regressor to use for initial predictions.

        increasing: bool, default=True
            If True, the isotonic regression will be monotonically increasing.
            If False, it will be monotonically decreasing.

        out_of_bounds: str, default='nan'
            The out_of_bounds parameter for IsotonicRegression.
            Can be 'nan', 'clip', or 'raise'.
    """

    def __init__(self, regr, increasing=True, out_of_bounds="nan"):
        """Initialize the IsotonicRegressor.

        Args:
            regr: estimator
                Base regressor to use for initial predictions.

            increasing: bool, default=True
                If True, the isotonic regression will be monotonically increasing.
                If False, it will be monotonically decreasing.

            out_of_bounds: str, default='nan'
                The out_of_bounds parameter for IsotonicRegression.
                Can be 'nan', 'clip', or 'raise'.
        """
        self.regr = regr
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def fit(self, X, y, **kwargs):
        """Fit the model.

        Args:
            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            **kwargs: additional parameters to be passed to the base regressor.

        Returns:
            self: object.
        """
        # Validate input
        X, y = check_X_y(X, y)
        # Fit the base regressor
        self.regr.fit(X, y, **kwargs)
        # Get predictions from base regressor for training data
        y_pred_base = self.regr.predict(X)
        # Fit isotonic regression on the base predictions vs actual targets
        self.isotonic_regressor_ = IsotonicRegression(
            increasing=self.increasing, out_of_bounds=self.out_of_bounds
        )
        self.isotonic_regressor_.fit(y_pred_base, y)
        return self

    def predict(self, X, **kwargs):
        """Predict using the model.

        Args:
            X: {array-like}, shape = [n_samples, n_features]
                Samples.

            **kwargs: additional parameters to be passed to the base regressor.

        Returns:
            y_pred: array-like, shape = [n_samples]
                Predicted values.
        """
        # Check if fitted
        check_is_fitted(self, ["regr", "isotonic_regressor_"])
        # Validate input
        X = check_array(X)
        # Get predictions from base regressor
        y_pred_base = self.regr.predict(X, **kwargs)
        # Apply isotonic regression postprocessing
        return self.isotonic_regressor_.predict(y_pred_base)
