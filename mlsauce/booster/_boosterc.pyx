# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

import functools
import numpy as np
cimport numpy as np
cimport cython
import gc

from cython.parallel cimport prange
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

DTYPE_int = np.int

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
 

# 1 - classifier ----- 

# 1 - 1 fit classifier ----- 

def fit_booster_classifier(double[:,::1] X, long int[:] y, 
                           int n_estimators=100, double learning_rate=0.1, 
                           int n_hidden_features=5, double reg_lambda=0.1, 
                           double row_sample=1, double col_sample=1,
                           double dropout=0, double tolerance=1e-4, 
                           int direct_link=1, int verbose=1,
                           int seed=123): 
  
  cdef long int n
  cdef int p, n_cols_h_i, n_cols_hh_i
  cdef int n_classes
  cdef Py_ssize_t iter
  cdef dict res
  cdef double ym
  cdef double[:] xm, xsd
  cdef double[:,::1] Y, X_, E, W_i, h_i, hh_i, hidden_layer_i, hhidden_layer_i
  
  
  n = X.shape[0]
  p = X.shape[1]
  res = {}
  
  xm = np.asarray(X).mean(axis=0)
  xsd = np.asarray(X).std(axis=0)
  
  res['direct_link'] = direct_link
  res['xm'] = np.asarray(xm)
  res['xsd'] = np.asarray(xsd)
  res['n_estimators'] = n_estimators
  res['learning_rate'] = learning_rate
  res['W_i'] = {}
  res['beta_i'] = {} # use res['ridge_obj'] instead (pickle obj)
  res['col_index_i'] = {}
  
  X_ = (np.asarray(X) - xm[None, :])/xsd[None, :]
  n_classes = len(np.unique(y))
  res['n_classes'] = n_classes
  
  Y = one_hot_encode(y, n_classes)
  Ym = np.mean(Y, axis=0)
  res['Ym'] = Ym
  E = Y - Ym
  iterator = tqdm(range(n_estimators)) if verbose else range(n_estimators)

  for iter in iterator:
      
      np.random.seed(seed + iter*1000)
    
      iy = np.sort(np.random.choice(a=range(p), 
                                    size=np.int(p*col_sample), 
                                    replace=False), 
                   kind='quicksort')
      res['col_index_i'][iter] = iy                     
      X_iy = np.asarray(X_)[:, iy] # must be X_!
      W_i = np.random.rand(X_iy.shape[1], n_hidden_features)
      hhidden_layer_i = dropout_func(x=np.maximum(np.dot(X_iy, W_i), 0), 
                                     drop_prob=dropout, seed=seed)
      hh_i = np.hstack((X_iy, hhidden_layer_i)) if direct_link else hhidden_layer_i
      
      if row_sample < 1:
      
        ix = np.sort(np.random.choice(a=range(n), 
                                    size=np.int(n*row_sample), 
                                    replace=False), 
                     kind='quicksort')
        X_iy_ix = X_iy[ix,:]       
        hidden_layer_i = dropout_func(x=np.maximum(np.dot(X_iy_ix, W_i), 0), 
                                      drop_prob=dropout, seed=seed)
        h_i =  np.hstack((X_iy_ix, hidden_layer_i)) if direct_link else hidden_layer_i
        beta_i = np.linalg.lstsq(a = np.vstack((h_i, sqrt(reg_lambda)*np.identity(h_i.shape[1]))), 
                                 b = np.vstack((np.asarray(E)[ix,:], np.zeros((h_i.shape[1], n_classes)))), 
                                 rcond = None)[0] 
      else:
      
        beta_i = np.linalg.lstsq(a = np.vstack((hh_i, sqrt(reg_lambda)*np.identity(hh_i.shape[1]))), 
                                 b = np.vstack((np.asarray(E), np.zeros((hh_i.shape[1], n_classes)))), 
                                 rcond = None)[0] 
      
      E -= learning_rate*np.dot(hh_i, beta_i) # use predict
      
      res['W_i'][iter] = np.asarray(W_i)
      
      res['beta_i'][iter] = beta_i # use res['ridge_obj'] instead (pickle obj)
      
      if np.linalg.norm(E, ord='fro') <= tolerance:
        res['n_estimators'] = iter
        break
      
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
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  n_row_preds = X.shape[0]
  preds_sum = np.zeros((n_row_preds, n_classes))
  out_probs = np.zeros((n_row_preds, n_classes))
  
  
  for iter in range(n_estimators):
  
    iy = obj['col_index_i'][iter]
    X_iy = X_[:, iy] # must be X_!
    W_i = obj['W_i'][iter]
    hh_i = np.hstack((X_iy, np.maximum(np.dot(X_iy, W_i), 0))) if direct_link else np.maximum(np.dot(X_iy, W_i), 0)
    preds_sum = preds_sum + learning_rate*np.dot(hh_i, obj['beta_i'][iter])
  
  out_probs = expit(np.tile(obj['Ym'], n_row_preds).reshape(n_row_preds, n_classes) + np.asarray(preds_sum))
  
  out_probs /= np.sum(out_probs, axis=1)[:, None]

  return np.asarray(out_probs)
  
  
# 2 - regressor -----   

# 2 - 1 fit regressor -----
  
def fit_booster_regressor(double[:,::1] X, double[:] y, 
                           int n_estimators=100, double learning_rate=0.1, 
                           int n_hidden_features=5, double reg_lambda=0.1, 
                           double row_sample=1, double col_sample=1,
                           double dropout=0, double tolerance=1e-4, 
                           int direct_link=1, int verbose=1, 
                           int seed=123): 
  
  cdef long int n
  cdef int p, n_cols_h_i, n_cols_hh_i
  cdef int n_classes
  cdef Py_ssize_t iter
  cdef dict res
  cdef double ym
  cdef double[:] xm, xsd, e
  cdef double[:,::1] X_, W_i, h_i, hh_i, hidden_layer_i, hhidden_layer_i
  
  
  n = X.shape[0]
  p = X.shape[1]
  res = {}
  
  xm = np.asarray(X).mean(axis=0)
  xsd = np.asarray(X).std(axis=0)
  
  res['direct_link'] = direct_link
  res['xm'] = np.asarray(xm)
  res['xsd'] = np.asarray(xsd)
  res['n_estimators'] = n_estimators
  res['learning_rate'] = learning_rate
  res['W_i'] = {}
  res['beta_i'] = {}
  res['col_index_i'] = {}
  
  X_ = (np.asarray(X) - xm[None, :])/xsd[None, :]
  n_classes = len(np.unique(y))
  res['n_classes'] = n_classes
  
  ym = np.mean(y)
  res['ym'] = ym
  e = y - np.repeat(ym, n)
  iterator = tqdm(range(n_estimators)) if verbose else range(n_estimators)

  for iter in iterator:
      
      np.random.seed(seed + iter*1000)
    
      iy = np.sort(np.random.choice(a=range(p), 
                                    size=np.int(p*col_sample), 
                                    replace=False), 
                   kind='quicksort')
      res['col_index_i'][iter] = iy                     
      X_iy = np.asarray(X_)[:, iy] # must be X_!
      W_i = np.random.rand(X_iy.shape[1], n_hidden_features)
      hhidden_layer_i = dropout_func(x=np.maximum(np.dot(X_iy, W_i), 0), 
                                     drop_prob=dropout, seed=seed)
      hh_i = np.hstack((X_iy, hhidden_layer_i)) if direct_link else hhidden_layer_i
      
      if row_sample < 1:
      
        ix = np.sort(np.random.choice(a=range(n), 
                                    size=np.int(n*row_sample), 
                                    replace=False), 
                     kind='quicksort')
        X_iy_ix = X_iy[ix,:]       
        hidden_layer_i = dropout_func(x=np.maximum(np.dot(X_iy_ix, W_i), 0), 
                                      drop_prob=dropout, seed=seed)
        h_i =  np.hstack((X_iy_ix, hidden_layer_i)) if direct_link else hidden_layer_i
        n_cols_h_i = h_i.shape[1]
        beta_i = np.linalg.lstsq(a = np.vstack((h_i, sqrt(reg_lambda)*np.identity(n_cols_h_i))), 
                                 b = np.concatenate((np.asarray(e)[ix], np.zeros((n_cols_h_i)))), 
                                 rcond = None)[0] 
      else:
      
        beta_i = np.linalg.lstsq(a = np.vstack((hh_i, sqrt(reg_lambda)*np.identity(hh_i.shape[1]))), 
                                 b = np.concatenate((np.asarray(e), np.zeros((hh_i.shape[1])))), 
                                 rcond = None)[0] 
      
      e -= learning_rate*np.dot(hh_i, beta_i)
      
      res['W_i'][iter] = np.asarray(W_i)
      
      res['beta_i'][iter] = beta_i
      
      if np.linalg.norm(e) <= tolerance:
        res['n_estimators'] = iter
        break
      
  return res
  
# 2 - 2 predict regressor -----

def predict_booster_regressor(object obj, double[:,::1] X):

  cdef int iter, n_estimators, n_classes
  cdef double learning_rate
  cdef double[:] preds_sum
  cdef double[:,::1] hh_i

  n_classes = obj['n_classes']
  direct_link = obj['direct_link']
  n_estimators = obj['n_estimators']
  learning_rate = obj['learning_rate']
  X_ = (X - obj['xm'][None, :])/obj['xsd'][None, :]
  preds_sum = np.zeros(X.shape[0])
  
  for iter in range(n_estimators):
  
    iy = obj['col_index_i'][iter]
    X_iy = X_[:, iy] # must be X_!
    W_i = obj['W_i'][iter]
    hh_i = np.hstack((X_iy, np.maximum(np.dot(X_iy, W_i), 0))) if direct_link else np.maximum(np.dot(X_iy, W_i), 0)
    
    preds_sum = preds_sum + learning_rate*np.dot(hh_i, obj['beta_i'][iter])
  
  return np.asarray(obj['ym'] + np.asarray(preds_sum))