import cython 
import numpy as np 
from typing import Any, Callable, Iterable
from math import factorial
from itertools import chain, combinations


@cython.cclass
class ShapExplainer():
    def __init__(self,
                 model: Callable[[np.ndarray], cython.float], 
                 background_dataset: np.ndarray,
                 max_samples: cython.int = -1):
        self.model = model

        # Ensure that background_dataset is valid and has a positive size
        if background_dataset.shape[0] == 0:
            raise ValueError("background_dataset cannot be empty")

        # If max_samples is -1, we use the full background dataset
        if max_samples == -1 or max_samples >= background_dataset.shape[0]:
            self.background_dataset = background_dataset
        else:
            # Ensure max_samples is non-negative and within the valid range
            if max_samples < 0:
                raise ValueError("max_samples must be non-negative")
            
            # Subsample the background dataset
            rng = np.random.default_rng()
            self.background_dataset = rng.choice(background_dataset, size=max_samples, replace=False)

    @cython.ccall
    def shap_values(self, X: np.ndarray): #-> np.ndarray:
        "SHAP Values for instances in DataFrame or 2D array"
        i: cython.int               # cdef int i
        j: cython.int               # cdef int i
        shap_values = np.empty(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                shap_values[i, j] = self._compute_single_shap_value(j, X[i, :])
        return shap_values

    @cython.cfunc      
    def _compute_single_shap_value(self, 
                                   feature: cython.int,
                                   instance: np.array) -> cython.float:
        "Compute a single SHAP value (equation 4)"
        n_features: cython.int               # cdef int n_features
        shap_value: cython.float             # cdef float shap_value
        n_features = len(instance)
        shap_value = 0
        for subset in self._get_all_other_feature_subsets(n_features, feature):
            n_subset = len(subset)
            prediction_without_feature = self._subset_model_approximation(
                subset, 
                instance
            )
            prediction_with_feature = self._subset_model_approximation(
                subset + (feature,), 
                instance
            )
            factor = self._permutation_factor(n_features, n_subset)
            shap_value += factor * (prediction_with_feature - prediction_without_feature)
        return shap_value
    
    def _get_all_subsets(self, items: list) -> Iterable:
        r: cython.int               # cdef int r
        return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))
    
    def _get_all_other_feature_subsets(self, n_features: cython.int, feature_of_interest: cython.int):
        return self._get_all_subsets(np.delete(np.arange(n_features), feature_of_interest).tolist())

    def _permutation_factor(self, n_features: cython.int, n_subset: cython.int):
        return (
            factorial(n_subset) 
            * factorial(n_features - n_subset - 1) 
            / factorial(n_features) 
        )
    
    def _subset_model_approximation(self, 
                                    feature_subset: tuple[cython.int, ...], 
                                    instance: np.array) -> cython.float:
        masked_background_dataset = self.background_dataset.copy()
        # https://mathspp.com/blog/breaking-out-of-nested-loops-with-generators
        j: cython.int               # cdef int j
        for j in range(masked_background_dataset.shape[1]):
            if j in feature_subset:
                masked_background_dataset[:, j] = instance[j]
        conditional_expectation_of_model = np.mean(
            self.model(masked_background_dataset)
        )
        return conditional_expectation_of_model          