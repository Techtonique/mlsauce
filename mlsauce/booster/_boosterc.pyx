# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

import functools
import pickle
import numpy as np
cimport numpy as np
cimport cython
import gc

from ..lasso import LassoRegressor
from ..ridge import RidgeRegressor
from ..elasticnet import ElasticNetRegressor
from libc.math cimport log, exp, sqrt
from numpy.linalg import lstsq
from numpy.linalg import norm
from scipy.special import expit
from tqdm import tqdm


# 0 - utils -----

# 0 - 1 data structures & funcs -----

# a tuple of doubles
cdef struct mydoubletuple:
    double elt1
    double elt2

DTYPE_double = np.double

DTYPE_int = np.int32

ctypedef np.double_t DTYPE_double_t


# 0 - 2 data structures & funcs -----

# dropout
def dropout_func(x, drop_prob=0, seed=123):

    assert 0 <= drop_prob <= 1

    n, p = x.shape

    if drop_prob == 0:
        return x

    if drop_prob == 1:
        return np.zeros_like(x)

    np.random.seed(seed)
    dropped_indices = np.random.rand(n, p) > drop_prob

    return dropped_indices * x / (1 - drop_prob)
    
# one-hot encoder for discrete response
def one_hot_encode(long int[:] y, 
                   int n_classes):
    
    cdef long int i 
    cdef long int n_obs = len(y)
    cdef double[:,::1] res = np.zeros((n_obs, n_classes), dtype=DTYPE_double)        

    for i in range(n_obs):
        res[i, y[i]] = 1

    return np.asarray(res)
 
# 0 - 3 activation functions ----- 

def relu_activation(x):
    return np.maximum(x, 0)

def relu6_activation(x):
    return np.minimum(np.maximum(x, 0), 6)

def sigmoid_activation(x):
    return 1/(1 + np.exp(-x))

def activation_choice(x):
  activation_options = {
                "relu": relu_activation,
                "relu6": relu6_activation,
                "tanh": np.tanh,
                "sigmoid": sigmoid_activation
            }
  return activation_options[x]  


# 1 - classifier ----- 

# 1 - 1 fit classifier ----- 

def fit_booster_classifier(double[:,::1] X, long int[:] y, 
                           int n_estimators=100, double learning_rate=0.1, 
                           int n_hidden_features=5, double reg_lambda=0.1, 
                           double alpha=0.5, 
                           double row_sample=1, double col_sample=1,
                           double dropout=0, double tolerance=1e-6, 
                           int direct_link=1, int verbose=1,
                           int seed=123, str backend="cpu", 
                           str solver="ridge", 
                           str activation="relu",
                           str weights_distr = "uniform",
                           object obj = None
                           ): 
  
  cdef long int n
  cdef int p
  cdef int n_classes
  cdef Py_ssize_t iter
  cdef dict res
  cdef double ym
  cdef double[:] xm, xsd
  cdef double[:,::1] Y, X_, E, W_i, h_i, hh_i, hidden_layer_i, hhidden_layer_i
  cdef double current_error
  
  
  n = X.shape[0]
  p = X.shape[1]
  res = {}
  current_error = 1000000.0
  
  xm = np.asarray(X).mean(axis=0)
  xsd = np.asarray(X).std(axis=0)
  for i in range(len(xsd)):
    if xsd[i] == 0:
      xsd[i] = 1.0 

  
  res['direct_link'] = direct_link
  res['xm'] = np.asarray(xm)
  res['xsd'] = np.asarray(xsd)
  res['n_estimators'] = n_estimators
  res['learning_rate'] = learning_rate
  res['W_i'] = {}
  res['fit_obj_i'] = {} 
  res['col_index_i'] = {}
  res['loss'] = []
  res['activation'] = activation
  res['weights_distr'] = weights_distr
  
  X_ = (np.asarray(X) - xm[None, :])/xsd[None, :]
  n_classes = len(np.unique(y))
  res['n_classes'] = n_classes
  
  Y = one_hot_encode(y, n_classes)
  Ym = np.mean(Y, axis=0)
  res['Ym'] = Ym
  E = Y - Ym
  iterator = tqdm(range(n_estimators)) if verbose else range(n_estimators)

  if obj is None: 
    if solver == "ridge":
      fit_obj = RidgeRegressor(reg_lambda = reg_lambda, backend = backend)
    elif solver == "lasso": 
      fit_obj = LassoRegressor(reg_lambda = reg_lambda, backend = backend)
    else: 
      fit_obj = ElasticNetRegressor(reg_lambda = reg_lambda, alpha = alpha, 
      backend = backend)  
  else: 
      fit_obj = obj 

  for iter in iterator:
      
      np.random.seed(seed + iter*1000)
    
      iy = np.sort(np.random.choice(a=range(p), 
                                    size=np.int32(p*col_sample), 
                                    replace=False), 
                   kind='quicksort')
      res['col_index_i'][iter] = iy                     
      X_iy = np.asarray(X_)[:, iy] # must be X_!
      if res['weights_distr' ]== "uniform":
        W_i = np.random.rand(X_iy.shape[1], n_hidden_features)
      else: 
        W_i = np.random.randn(X_iy.shape[1], n_hidden_features)
      hhidden_layer_i = dropout_func(x=activation_choice(activation)(np.dot(X_iy, W_i)), 
                                     drop_prob=dropout, seed=seed)
      hh_i = np.hstack((X_iy, hhidden_layer_i)) if direct_link else hhidden_layer_i
      
      if row_sample < 1:
      
        ix = np.sort(np.random.choice(a=range(n), 
                                    size=np.int32(n*row_sample), 
                                    replace=False), 
                     kind='quicksort')
        X_iy_ix = X_iy[ix,:]       
        hidden_layer_i = dropout_func(x=activation_choice(activation)(np.dot(X_iy_ix, W_i)), 
                                      drop_prob=dropout, seed=seed)
        h_i =  np.hstack((X_iy_ix, hidden_layer_i)) if direct_link else hidden_layer_i
        fit_obj.fit(X = np.asarray(h_i), y = np.asarray(E)[ix,:])
                                 
      else:
      
        fit_obj.fit(X = np.asarray(hh_i), y = np.asarray(E))
            
      E = E - learning_rate*np.asarray(fit_obj.predict(np.asarray(hh_i)))
      
      res['W_i'][iter] = np.asarray(W_i)
            
      res['fit_obj_i'][iter] = pickle.loads(pickle.dumps(fit_obj, -1))

      current_error = np.linalg.norm(E, ord='fro')

      res['loss'].append(current_error)
      
      try:
        if np.abs(np.flip(np.diff(res['loss'])))[0] <= tolerance:
          res['n_estimators'] = iter
          break
      except:
        pass
      
  return res
  
  
# 1 - 2 predict classifier ----- 

def predict_proba_booster_classifier(object obj, double[:,::1] X):

  cdef int iter, n_estimators, n_classes
  cdef double learning_rate
  cdef double[:,::1] preds_sum, out_probs
  cdef long int n_row_preds 
  

  n_classes = obj['n_classes']
  direct_link = obj['direct_link']
  n_estimators = obj['n_estimators']
  learning_rate = obj['learning_rate']
  activation = obj['activation']
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  n_row_preds = X.shape[0]
  preds_sum = np.zeros((n_row_preds, n_classes))
  out_probs = np.zeros((n_row_preds, n_classes))
  
  
  for iter in range(n_estimators):
  
    iy = obj['col_index_i'][iter]
    X_iy = X_[:, iy] # must be X_!
    W_i = obj['W_i'][iter]
    hh_i = np.hstack((X_iy, activation_choice(activation)(np.dot(X_iy, W_i)))) if direct_link else activation_choice(activation)(np.dot(X_iy, W_i))
    preds_sum = preds_sum + learning_rate*np.asarray(obj['fit_obj_i'][iter].predict(np.asarray(hh_i)))
  
  out_probs = expit(np.tile(obj['Ym'], n_row_preds).reshape(n_row_preds, n_classes) + np.asarray(preds_sum))
  
  out_probs = out_probs/np.sum(out_probs, axis=1)[:, None]

  return np.asarray(out_probs)
  
  
# 2 - regressor -----   

# 2 - 1 fit regressor -----
  
def fit_booster_regressor(double[:,::1] X, double[:] y, 
                           int n_estimators=100, double learning_rate=0.1, 
                           int n_hidden_features=5, double reg_lambda=0.1, 
                           double alpha=0.5, 
                           double row_sample=1, double col_sample=1,
                           double dropout=0, double tolerance=1e-6, 
                           int direct_link=1, int verbose=1, 
                           int seed=123, str backend="cpu", 
                           str solver="ridge", str activation="relu", 
                           str weights_distr = "uniform",
                           object obj = None): 
  
  cdef long int n
  cdef int i, p
  cdef int n_classes 
  cdef Py_ssize_t iter
  cdef dict res
  cdef double ym
  cdef double[:] xm, xsd, e
  cdef double[:,::1] X_, W_i, h_i, hh_i, hidden_layer_i, hhidden_layer_i
  cdef double current_error
  
  
  n = X.shape[0]
  p = X.shape[1]
  res = {}
  current_error = 1000000.0
  
  xm = np.asarray(X).mean(axis=0)
  xsd = np.asarray(X).std(axis=0)
  for i in range(len(xsd)):
    if xsd[i] == 0:
      xsd[i] = 1.0 

  
  res['direct_link'] = direct_link
  res['xm'] = np.asarray(xm)
  res['xsd'] = np.asarray(xsd)
  res['n_estimators'] = n_estimators
  res['learning_rate'] = learning_rate
  res['activation'] = activation
  res['W_i'] = {}
  res['fit_obj_i'] = {} 
  res['col_index_i'] = {}
  res['loss'] = []
  res['weights_distr'] = weights_distr
  res['n_obs'] = n 
  
  X_ = (np.asarray(X) - xm[None, :])/xsd[None, :]
  n_classes = len(np.unique(y))
  res['n_classes'] = n_classes
  
  ym = np.mean(y)
  res['ym'] = ym
  e = y - np.repeat(ym, n)
  iterator = tqdm(range(n_estimators)) if verbose else range(n_estimators)

  if obj is None: 
    if solver == "ridge":
      fit_obj = RidgeRegressor(reg_lambda = reg_lambda, backend = backend)
    elif solver == "lasso": 
      fit_obj = LassoRegressor(reg_lambda = reg_lambda, backend = backend)
    else: 
      fit_obj = ElasticNetRegressor(reg_lambda = reg_lambda, alpha = alpha, 
      backend = backend)  
  else:
      fit_obj = obj 

  for iter in iterator:
      
      np.random.seed(seed + iter*1000)
    
      iy = np.sort(np.random.choice(a=range(p), 
                                    size=np.int32(p*col_sample), 
                                    replace=False), 
                   kind='quicksort')
      res['col_index_i'][iter] = iy                     
      X_iy = np.asarray(X_)[:, iy] # must be X_!
      if res['weights_distr' ] == "uniform":
        W_i = np.random.rand(X_iy.shape[1], n_hidden_features)
      else: 
        W_i = np.random.randn(X_iy.shape[1], n_hidden_features)
      hhidden_layer_i = dropout_func(x=activation_choice(activation)(np.dot(X_iy, W_i)), 
                                     drop_prob=dropout, seed=seed)
      hh_i = np.hstack((X_iy, hhidden_layer_i)) if direct_link else hhidden_layer_i
      
      if row_sample < 1:
      
        ix = np.sort(np.random.choice(a=range(n), 
                                    size=np.int32(n*row_sample), 
                                    replace=False), 
                     kind='quicksort')
        X_iy_ix = X_iy[ix,:]       
        hidden_layer_i = dropout_func(x=activation_choice(activation)(np.dot(X_iy_ix, W_i)), 
                                      drop_prob=dropout, seed=seed)
        h_i =  np.hstack((X_iy_ix, hidden_layer_i)) if direct_link else hidden_layer_i        
        fit_obj.fit(X = np.asarray(h_i), y = np.asarray(e)[ix])

      else:
      
        fit_obj.fit(X = np.asarray(hh_i), y = np.asarray(e))
            
      e = e - learning_rate*np.asarray(fit_obj.predict(np.asarray(hh_i)))

      res['W_i'][iter] = np.asarray(W_i)
      
      res['fit_obj_i'][iter] = pickle.loads(pickle.dumps(fit_obj, -1))

      current_error = np.linalg.norm(e)

      res['loss'].append(current_error)      

      try:              
        if np.abs(np.flip(np.diff(res['loss'])))[0] <= tolerance:
          res['n_estimators'] = iter
          break      
      except:
        pass
      
  return res
  
# 2 - 2 predict regressor -----

def predict_booster_regressor(object obj, double[:,::1] X):

  cdef int iter, n_estimators, n_classes
  cdef double learning_rate
  cdef double[:] preds_sum
  cdef double[:,::1] hh_i

  direct_link = obj['direct_link']
  n_estimators = obj['n_estimators']
  learning_rate = obj['learning_rate']
  activation = obj['activation']
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  preds_sum = np.zeros(X.shape[0])
  
  for iter in range(n_estimators):
  
    iy = obj['col_index_i'][iter]
    X_iy = X_[:, iy] # must be X_!
    W_i = obj['W_i'][iter]
    hh_i = np.hstack((X_iy, activation_choice(activation)(np.dot(X_iy, W_i)))) if direct_link else activation_choice(activation)(np.dot(X_iy, W_i))        
    preds_sum = preds_sum + learning_rate*np.asarray(obj['fit_obj_i'][iter].predict(np.asarray(hh_i)))
  
  return np.asarray(obj['ym'] + np.asarray(preds_sum))

# 2 - 3 update regressor -----

def update_booster_regressor(object obj, double[:] X, double y, double alpha=0.5):

  cdef int iter, n_estimators, n_classes, n_obs
  cdef double learning_rate, preds_sum, residuals_i, centered_y
  cdef double[:] xm_old
  cdef double[:,::1] hh_i
  preds_sum = 0

  n_obs = obj['n_obs']
  direct_link = obj['direct_link']
  n_estimators = obj['n_estimators']
  learning_rate = obj['learning_rate']
  activation = obj['activation']
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  centered_y = y - obj['ym']
  
  for iter in range(n_estimators):
  
    iy = obj['col_index_i'][iter]
    X_iy = np.asarray(X_[:, iy]).reshape(1, -1) # must be X_!
    W_i = obj['W_i'][iter]
    hh_i = np.hstack((X_iy, activation_choice(activation)(np.dot(X_iy, W_i)))) if direct_link else activation_choice(activation)(np.dot(X_iy, W_i))        
    preds_sum = preds_sum + learning_rate*np.asarray(obj['fit_obj_i'][iter].predict(np.asarray(hh_i)))
    residuals_i = centered_y - preds_sum
    obj['fit_obj_i'][iter].coef_ = np.asarray(obj['fit_obj_i'][iter].coef_).ravel() + (n_obs**(-alpha))*np.dot(residuals_i, hh_i).ravel()    
    
  xm_old = obj['xm']
  obj['xm'] = (n_obs*np.asarray(xm_old) + X)/(n_obs + 1)
  obj['ym'] = (n_obs*obj['ym'] + y)/(n_obs + 1)
  obj['xsd'] = np.sqrt(((n_obs - 1)*(obj['xsd']**2) + (np.asarray(X) -np.asarray(xm_old))*(np.asarray(X) - obj['xm']))/n_obs)  
  obj['n_obs'] = n_obs + 1
  
  return obj
