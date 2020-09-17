# Classifiers 

_In alphabetical order_

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/adaopt/_adaopt.py#L13)</span>

### AdaOpt


```python
mlsauce.AdaOpt(
    n_iterations=50,
    learning_rate=0.3,
    reg_lambda=0.1,
    reg_alpha=0.5,
    eta=0.01,
    gamma=0.01,
    k=3,
    tolerance=0,
    n_clusters=0,
    batch_size=100,
    row_sample=0.8,
    type_dist="euclidean-f",
    n_jobs=None,
    verbose=0,
    cache=True,
    seed=123,
)
```


AdaOpt classifier.
    
Attributes:    

    n_iterations: int
        number of iterations of the optimizer at training time.

    learning_rate: float
        controls the speed of the optimizer at training time.  

    reg_lambda: float
        L2 regularization parameter for successive errors in the optimizer 
        (at training time).

    reg_alpha: float
        L1 regularization parameter for successive errors in the optimizer 
        (at training time).

    eta: float
        controls the slope in gradient descent (at training time).

    gamma: float
        controls the step size in gradient descent (at training time).

    k: int
        number of nearest neighbors selected at test time for classification.

    tolerance: float
        controls early stopping in gradient descent (at training time). 

    n_clusters: int
        number of clusters, if MiniBatch k-means is used at test time 
        (for faster prediction).

    batch_size: int
        size of the batch, if MiniBatch k-means is used at test time 
        (for faster prediction).

    row_sample: float
        percentage of rows chosen from training set (by stratified subsampling, 
        for faster prediction).

    type_dist: str
        distance used for finding the nearest neighbors; currently `euclidean-f` 
        (euclidean distances calculated as whole), `euclidean` (euclidean distances 
        calculated row by row), `cosine` (cosine distance).

    n_jobs: int 
        number of cpus for parallel processing (default: None)

    verbose: int
        progress bar for parallel processing (yes = 1) or not (no = 0)

    cache: boolean
        if the nearest neighbors are cached or not, for faster retrieval in 
        subsequent calls.

    seed: int 
        reproducibility seed for nodes_sim=='uniform', clustering and dropout.
     


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/adaopt/_adaopt.py#L120)</span>

### fit


```python
AdaOpt.fit(X, y, **kwargs)
```


Fit AdaOpt to training data (X, y)

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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/adaopt/_adaopt.py#L179)</span>

### predict


```python
AdaOpt.predict(X, **kwargs)
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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/adaopt/_adaopt.py#L198)</span>

### predict_proba


```python
AdaOpt.predict_proba(X, **kwargs)
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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/booster/_booster_classifier.py#L10)</span>

### LSBoostClassifier


```python
mlsauce.LSBoostClassifier(
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


LSBoost classifier.
    
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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/booster/_booster_classifier.py#L108)</span>

### fit


```python
LSBoostClassifier.fit(X, y, **kwargs)
```


Fit Booster (classifier) to training data (X, y)

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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/booster/_booster_classifier.py#L147)</span>

### predict


```python
LSBoostClassifier.predict(X, **kwargs)
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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/booster/_booster_classifier.py#L166)</span>

### predict_proba


```python
LSBoostClassifier.predict_proba(X, **kwargs)
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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/stump/_stump_classifier.py#L7)</span>

### StumpClassifier


```python
mlsauce.StumpClassifier(bins="auto")
```


Stump classifier.
    
Attributes:

    bins: int
        Number of histogram bins; as in numpy.histogram.        


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/stump/_stump_classifier.py#L21)</span>

### fit


```python
StumpClassifier.fit(X, y, sample_weight=None, **kwargs)
```


Fit Stump to training data (X, y)

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


----

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/stump/_stump_classifier.py#L60)</span>

### predict


```python
StumpClassifier.predict(X, **kwargs)
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

<span style="float:right;">[[source]](https://github.com/Techtonique/mlsauce/mlsauce/stump/_stump_classifier.py#L79)</span>

### predict_proba


```python
StumpClassifier.predict_proba(X, **kwargs)
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

