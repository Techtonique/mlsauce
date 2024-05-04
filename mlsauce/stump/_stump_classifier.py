import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

try:
    from . import _stumpc as stumpc
except ImportError:
    import _stumpc as stumpc


class StumpClassifier(BaseEstimator, ClassifierMixin):
    """Stump classifier.

    Attributes:

        bins: int
            Number of histogram bins; as in numpy.histogram.
    """

    def __init__(self, bins="auto"):
        self.bins = bins
        self.obj = None

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit Stump to training data (X, y)

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            sample_weight: array_like, shape = [n_samples]
                Observations weights.

        Returns:

            self: object.
        """

        if sample_weight is None:
            self.obj = stumpc.fit_stump_classifier(
                X=np.asarray(X, order="C"),
                y=np.asarray(y, order="C"),
                bins=self.bins,
            )

            return self

        self.obj = stumpc.fit_stump_classifier(
            X=np.asarray(X, order="C"),
            y=np.asarray(y, order="C"),
            sample_weight=np.ravel(sample_weight, order="C"),
            bins=self.bins,
        )
        self.n_classes_ = len(np.unique(y))  # for compatibility with sklearn
        return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to `predict_proba`


        Returns:

            model predictions: {array-like}
        """

        return np.argmax(self.predict_proba(X, **kwargs), axis=1)

    def predict_proba(self, X, **kwargs):
        """Predict probabilities for test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                self.cook_test_set

        Returns:

            probability estimates for test data: {array-like}
        """

        return stumpc.predict_proba_stump_classifier(
            self.obj, np.asarray(X, order="C")
        )
