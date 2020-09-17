# Regressors

_In alphabetical order_

All models possess methods `fit`, `predict`, and `score`. Methods `predict` and `score` are only documented for the first model; the __same principles apply subsequently__. For scoring metrics, refer to [scoring metrics](scoring_metrics.md). 

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/lasso/_lasso.py#L16)</span>

### LassoRegressor


```python
mlsauce.LassoRegressor(reg_lambda=0.1, max_iter=10, tol=0.001, backend="cpu")
```


Lasso.
    
Attributes:

    reg_lambda: float
        L1 regularization parameter.

    max_iter: int
        number of iterations of lasso shooting algorithm.

    tol: float          
        tolerance for convergence of lasso shooting algorithm.

    backend: str    
        type of backend; must be in ('cpu', 'gpu', 'tpu').


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/lasso/_lasso.py#L56)</span>

### fit


```python
LassoRegressor.fit(X, y, **kwargs)
```


Fit matrixops (classifier) to training data (X, y)

Args: 

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number 
        of samples and n_features is the number of features.
    
    y: array-like, shape = [n_samples]
        Target values.

    **kwargs: additional parameters to be passed to self.cook_training_set.
       
Returns:

    self: object.


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/lasso/_lasso.py#L138)</span>

### predict


```python
LassoRegressor.predict(X, **kwargs)
```


Predict test data X.

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number 
        of samples and n_features is the number of features.

    **kwargs: additional parameters to be passed to `predict_proba`
        
       
Returns:

    model predictions: {array-like}
    


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/booster/_booster_regressor.py#L10)</span>

### LSBoostRegressor


```python
mlsauce.LSBoostRegressor(
    n_estimators=100,
    learning_rate=0.1,
    n_hidden_features=5,
    reg_lambda=0.1,
    row_sample=1,
    col_sample=1,
    dropout=0,
    tolerance=0.0001,
    direct_link=1,
    verbose=1,
    seed=123,
    backend="cpu",
    solver="ridge",
)
```


LSBoost regressor.
    
Attributes: 
 
    n_estimators: int
        number of boosting iterations.

    learning_rate: float
        controls the learning speed at training time.  

    n_hidden_features: int 
        number of nodes in successive hidden layers.

    reg_lambda: float
        L2 regularization parameter for successive errors in the optimizer 
        (at training time).

    row_sample: float
        percentage of rows chosen from the training set.

    col_sample: float
        percentage of columns chosen from the training set.

    dropout: float
        percentage of nodes dropped from the training set. 

    tolerance: float
        controls early stopping in gradient descent (at training time).

    direct_link: bool
        indicates whether the original features are included (True) in model's 
        fitting or not (False).

    verbose: int
        progress bar (yes = 1) or not (no = 0) (currently).

    seed: int 
        reproducibility seed for nodes_sim=='uniform', clustering and dropout.

    backend: str    
        type of backend; must be in ('cpu', 'gpu', 'tpu') 

    solver: str    
        type of 'weak' learner; currently in ('ridge', 'lasso')         


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/booster/_booster_regressor.py#L109)</span>

### fit


```python
LSBoostRegressor.fit(X, y, **kwargs)
```


Fit Booster (regressor) to training data (X, y)

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number 
        of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
       Target values.

    **kwargs: additional parameters to be passed to self.cook_training_set.
       
Returns:

    self: object.


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/booster/_booster_regressor.py#L148)</span>

### predict


```python
LSBoostRegressor.predict(X, **kwargs)
```


Predict probabilities for test data X.

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number 
        of samples and n_features is the number of features.

    **kwargs: additional parameters to be passed to 
        self.cook_test_set
       
Returns:

    probability estimates for test data: {array-like}        


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/ridge/_ridge.py#L15)</span>

### RidgeRegressor


```python
mlsauce.RidgeRegressor(reg_lambda=0.1, backend="cpu")
```


Ridge.
    
Attributes:
 
    reg_lambda: float
        regularization parameter.

    backend: str    
        type of backend; must be in ('cpu', 'gpu', 'tpu') 
                 


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/ridge/_ridge.py#L47)</span>

### fit


```python
RidgeRegressor.fit(X, y, **kwargs)
```


Fit matrixops (classifier) to training data (X, y)

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number 
        of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
        Target values.

    **kwargs: additional parameters to be passed to self.cook_training_set.
       
Returns:

    self: object.


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/ridge/_ridge.py#L88)</span>

### predict


```python
RidgeRegressor.predict(X, **kwargs)
```


Predict test data X.

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number 
        of samples and n_features is the number of features.

    **kwargs: additional parameters to be passed to `predict_proba`
                       
Returns: 

    model predictions: {array-like}


----

