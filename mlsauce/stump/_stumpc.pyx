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
import operator
import warnings
import pickle

from collections import Counter
from cython.parallel cimport prange
from libc.math cimport log, exp, sqrt, fabs
from numpy.linalg import lstsq
from numpy.linalg import norm
from scipy.special import expit
from ..utils import safe_sparse_dot


# 0 - utils -----

warnings.filterwarnings("ignore")

# returns min(x, y)
cdef double min_c(double x, double y):
    if (x < y):
        return x 
    return y

# returns max(x, y)
cdef double max_c(double x, double y):
    if (x > y):
        return x 
    return y


# 0 - 1 data structures & funcs -----

# a tuple of doubles
cdef struct mydoubletuple:
    double elt1
    double elt2

ctypedef fused nparray_int:
    int
    
ctypedef fused nparray_double:    
    double
    
ctypedef fused nparray_long:
    long long


# 0 - 2 data structures & funcs -----

# one-hot encoder for discrete response
def one_hot_encode(long int[:] y, 
                   int n_classes):
    
    cdef long int i 
    cdef long int n_obs = len(y)
    cdef double[:,::1] res = np.zeros((n_obs, n_classes))        

    for i in range(n_obs):
        res[i, y[i]] = 1

    return np.asarray(res)

# average stuff
cdef double cython_average(double[:] x, double[:] weights):
    
    cdef long int i, n 
    cdef double sum_x    
    cdef double[:] sum_out 
    
    n = len(x)
    sum_x = 0 

    for i in range(n):        
      sum_x += x[i]*weights[i]        
    
    return sum_x

# obtain histogram bins        
def histogram_bins(double[:] x, long int n_bins = -1):
    
    cdef long int i, n    
    cdef long int n_classes, n_bins_1 
    cdef double freq     
    cdef double[:] res
    cdef double min_x, max_x

    n = len(x)    

    if n_bins == -1:
      n_bins = int(sqrt(n))    

    min_x = np.min(x)    
    max_x = np.max(x)

    freq = (max_x - min_x)/n_bins
    
    n_bins_1 = n_bins + 1

    res = np.zeros(n_bins_1)

    for i in range(n_bins_1):

      res[i] = min_x + i*freq
                                
    return np.asarray(res)

   
# 1 - fitting -----    


def fit_stump_classifier(double[:,::1] X, long int[:] y, 
                         double[:] sample_weight=None, 
                         bins="auto"):

  cdef long int n
  cdef long int n_up 
  cdef long int n_down 
  cdef long int n_preds
  cdef int n_bins 
  cdef int p
  cdef long int i
  cdef int j, k
  cdef int best_col
  cdef double[:] cutpoints
  cdef double cutpoint_i
  cdef double best_cutpoint
  cdef long int n_cutpoints
  cdef double error_rate, error_rate_cur, error_rate_down, error_rate_up
  cdef int best_class, class_up, n_classes
  cdef dict res_class
  cdef double[:] equal_weights


  n = X.shape[0]
  p = X.shape[1]
  n_classes = len(np.unique(y))
  res_class = {}
  error_rate = 10000.0
  error_rate_cur = 10000.0
  best_cutpoint = 0.0
  i = 0
  j = 0
  k = 0
  if bins == "auto":
    n_bins = -1
  else: 
    n_bins = bins  
  n_preds = len(y)  
  equal_weights = np.repeat(1/n_preds, n_preds)
  
  X_ = np.asarray(X).T.tolist()
  
  if n_classes <= 2:    
  
    for j in range(p):
    
      X_j = np.asarray(X_[j])
      
      cutpoints = histogram_bins(X_j, n_bins) 

      n_cutpoints = len(cutpoints)
      
      for i in range(n_cutpoints):
               
        cutpoint_i = cutpoints[i] 
        index_up = (X_j <= cutpoint_i)
        y_up = np.asarray(y)[index_up]        
        class_up = Counter(y_up.tolist()).most_common(1)[0][0] # majority vote # majority vote

        preds = class_up*index_up + (1 - class_up)*np.logical_not(index_up)                
        
        error_rate_cur = cython_average((preds != y)*1.0, equal_weights) if sample_weight is None else cython_average((preds != y)*1.0, sample_weight)           
        
        if error_rate_cur <= error_rate:
          # print(error_rate_cur)
          best_col = j
          best_cutpoint = cutpoint_i
          error_rate = error_rate_cur
          best_class = class_up
         
    return best_col, best_cutpoint, best_class

# if n_classes > 2:

  Y = one_hot_encode(y, n_classes)
  Y_ = Y.T.tolist()
  
  for k in range(n_classes):
  
    y_ = np.asarray(Y_[k])
    best_col = 0
    best_cutpoint = 0.0
    best_class = 0
    error_rate = 10000.0
    error_rate_cur = 10000.0
    best_cutpoint = 0.0
  
    for j in range(p):
    
      X_j = np.asarray(X_[j])
           
      cutpoints = histogram_bins(X_j, n_bins) 

      n_cutpoints = len(cutpoints)      
 
      for i in range(n_cutpoints):
        
        try:
        
          cutpoint_i = cutpoints[i] 
          index_up = (X_j <= cutpoint_i)
          y_up = y_[index_up]          
          class_up = Counter(y_up.tolist()).most_common(1)[0][0] # majority vote
  
          preds = class_up*index_up + (1 - class_up)*np.logical_not(index_up)
                     
          error_rate_cur = cython_average((preds != y_)*1.0, equal_weights) if sample_weight is None else cython_average((preds != y_)*1.0, sample_weight)                     
          
          if error_rate_cur <= error_rate:
            # print(error_rate_cur)
            best_col = j
            best_cutpoint = cutpoint_i
            error_rate = error_rate_cur
            best_class = class_up
  
        except:
        
          pass
    
    res_class[k] = (best_col, best_cutpoint, best_class)
        
  return res_class
  
  
  
# 2 - predict -----   

def predict_proba_twoclass(double[:,::1] X, 
                             int best_col, 
                             double best_cutpoint, 
                             int best_class):
    X_ = np.asarray(X).T.tolist()
    X_j = np.asarray(X_[best_col])
    preds = (X_j <= best_cutpoint)*best_class + (X_j > best_cutpoint)*(1 - best_class)
    return (np.asarray(one_hot_encode(preds, 2)))

def predict_proba_stump_classifier(object obj, double[:,::1] X):
  
  cdef long int n_obs
  cdef int n_classes
  cdef double[:,:] raw_probs
  cdef long int[:] temp_prob


  # if n_classes <= 2  
  if (isinstance(obj, dict) == False):
    return (predict_proba_twoclass(X, obj[0], obj[1], obj[2]))
    
    
  # else: n_classes > 2
  n_obs = X.shape[0]
  n_classes = len(obj)
  raw_probs = np.zeros((n_obs, n_classes))
  
  
  for k in range(n_classes):
    # print(f"class {k}")
    # print(predict_proba_twoclass(X, obj[k][0], obj[k][1], obj[k][2]))
    temp_prob = np.argmax(predict_proba_twoclass(X, obj[k][0], obj[k][1], obj[k][2]), 
                              axis=1)
    for i in range(n_obs):
      raw_probs[i, k] = temp_prob[i]
  
  # print("raw_probs")
  # print(np.asarray(raw_probs))
  expit_raw_probs = expit(raw_probs)
  
  return (expit_raw_probs/expit_raw_probs.sum(axis=1)[:, None])
  
  
  
    
  