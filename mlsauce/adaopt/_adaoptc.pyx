# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: Thierry Moudiki
#
# License: BSD 3

import functools
import numpy as np
cimport numpy as np
cimport cython
import gc

from cython.parallel cimport prange
from libc.math cimport log, exp, sqrt, fabs
from numpy.linalg import lstsq
from numpy.linalg import norm
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm

try:
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv
except ImportError:
    pass

from ..utils import subsample


# 0 - utils -----


# 0 - 0 data structures & funcs -----

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

cdef dict __find_kmin_x_cache = {}
    


# 0 - 1 fitting -----

# returns max(x, y)
cdef double max_c(double x, double y):
    if (x > y):
        return x 
    return y


# returns min(x, y)
cdef double min_c(double x, double y):
    if (x < y):
        return x 
    return y


# sum vector 
cdef double cython_sum(double[:] x, long int n):
    cdef double res = 0 
    cdef long int i
    for i in range(n):
        res += x[i]
    return res


# finds index of a misclassed element 
cdef long int find_misclassed_index(nparray_long[:] misclass, 
                                    long int n_obs):    
    
    cdef long int i  
    for i in range(n_obs):        
        if (misclass[i] != 1000): # a badly classified example            
            return i    
    return 100000


# calculates sumproduct or 2 vectors
cdef double sum_product(long int[:] x, double[:] y, 
                        long int n):    
    cdef double res = 0
    cdef long int i    
    for i in range(n):
        if (x[i] != 0) & (y[i] != 0):
            res += x[i]*y[i]    
    return res


# calculates tolerance for early stopping
cdef double calculate_tolerance(nparray_double[:] x):     
    
    cdef long int n    
    cdef double ans 
    cdef nparray_double[:] temp
    
    if (len(x) >= 5): 
        
        x_ = np.asarray(x)                
        temp = x_[np.where(x_ != 0.0)] 
        n = len(temp)
        
        try:
            
            ans = (temp[n-1] - temp[(n-2)])        
            if (ans > 0):
                return ans
            return -ans
        
        except:
            
            return 1e5
        
    else:
        
        return 1e5
        
 
# calculates updating factor for misclassified example prob
cdef mydoubletuple calculate_diff_proba(long int n_obs, 
                         nparray_long[:] misclass,
                         nparray_double[:] w_new, 
                         nparray_double[:] w_prev,                          
                         double v_prev,
                         double eta,
                         double gamma):        
    
    cdef double v_new
    cdef double w_new_
    cdef double w_prev_
    cdef long int misclass_index
    
    misclass_index = find_misclassed_index(misclass, 
                                           n_obs)        
    
    if misclass_index != 1e5: 
    
        w_new_ = max_c(w_new[misclass_index], 
                       2.220446049250313e-16) 
            
        w_prev_ = w_prev[misclass_index]   
                
        v_new = (w_new_ - w_prev_)
                                                          
        return mydoubletuple(-(v_prev*gamma - v_new)/eta, v_new)                                                     
    
    return mydoubletuple(0, 0)
    

# updates probs
def update_proba(double[:,::1] probs, 
                 nparray_long[:] misclass,
                 nparray_double[:] w_new, 
                 nparray_double[:] w_prev,                 
                 double v_prev, double eta, double gamma, 
                 long int n_obs, int n_classes):        
    
    cdef double[:,::1] new_probs = np.asarray(probs) # contains result    
    cdef double diff_proba, new_p, prob    
    cdef long int idx    
    cdef int idy
    cdef double v_new
    
    
    res = calculate_diff_proba(n_obs, misclass, 
                               w_new, w_prev, v_prev, 
                               eta, gamma)     

    diff_proba = res.elt1
    v_new = res.elt2
                
    # loop on all examples 
    for idx in range(n_obs):
        
        if misclass[idx] != 1000: # a badly classified example                                                              
                        
            new_probs[idx, misclass[idx]] -= diff_proba # update proba
            # DO NOT normalize row (!)                         
    
    return tuple((new_probs, v_new))


# one-hot encoder for discrete response
def one_hot_encode(long int[:] y, 
                   int n_classes):
    
    cdef long int i 
    cdef long int n_obs = len(y)
    cdef double[:,::1] res = np.zeros((n_obs, n_classes))        

    for i in range(n_obs):
        res[i, y[i]] = 1

    return np.asarray(res)


# 0 - 2 predict -----

# squared norms
# @Paul Panzer's soln
# from https://stackoverflow.com/questions/42991347/how-to-find-the-pairwise-differences-between-rows-of-two-very-large-matrices-usi    
def outer_sum_dot(A, B, backend="cpu"):
    if backend != "cpu":
        device_put(A)
        device_put(B)
    return np.add.outer((A*A).sum(axis=-1), (B*B).sum(axis=-1)) - 2*(A @ B.T)
    

cdef double norm_c(double[:] x, long int n) nogil:
    
    cdef double res = 0
    
    for i in range(n):        
        res += x[i]**2
        
    return(sqrt(res))
    

cdef double euclidean_distance_c(double[:] x, 
                                 double[:] y,  
                                 long int n) nogil:
    
    cdef double res = 0
    
    for i in range(n):        
        res += (x[i] - y[i])**2
    
    return(sqrt(res))


cdef double cosine_similarity_c(double[:] x, 
                                double[:] y,  
                                long int n) nogil:
    
    cdef double dotprod = 0
    
    for i in range(n):        
        dotprod += (x[i]*y[i])**2
    
    return(1 - dotprod/(norm_c(x, n)*norm_c(y, n)))    
    

# distance of vector to matrix rows
# keep numpy arrays x, B    
cdef double manhattan_distance_c(double[:] x, 
                                 double[:] y,  
                                 long int n) nogil:
    
    cdef double res = 0
    
    for i in range(n):        
        res += fabs(x[i] - y[i])
    
    return(res)
    
    
# distance of vector to matrix rows
# keep numpy arrays x, B    
cdef nparray_double[:] distance_to_mat_euclidean(nparray_double[:] x, 
                   nparray_double[:,:] B):
    
    cdef long int i
    cdef long int n_B = B.shape[0]
    cdef long int p_B = B.shape[1]
    cdef nparray_double[:] res = np.zeros(n_B)
    

    for i in prange(n_B, nogil=True):

        res[i] = euclidean_distance_c(x, B[i, :], p_B)
                                
    return np.asarray(res)
    

# distance of vector to matrix rows
# keep numpy arrays x, B    
cdef nparray_double[:] distance_to_mat_manhattan(nparray_double[:] x, 
                   nparray_double[:,:] B):
    
    cdef long int i
    cdef long int n_B = B.shape[0]
    cdef long int p_B = B.shape[1]
    cdef nparray_double[:] res = np.zeros(n_B)
    

    for i in prange(n_B, nogil=True):

        res[i] = manhattan_distance_c(x, B[i, :], p_B)
                                
    return np.asarray(res)


# distance of vector to matrix rows
# keep numpy arrays x, B    
cdef nparray_double[:] distance_to_mat_cosine(nparray_double[:] x, 
                   nparray_double[:,:] B):
    
    cdef long int i
    cdef long int n_B = B.shape[0]
    cdef long int p_B = B.shape[1]
    cdef nparray_double[:] res = np.zeros(n_B)
    

    for i in prange(n_B, nogil=True):

        res[i] = cosine_similarity_c(x, B[i, :], p_B)        
                            
    return np.asarray(res)


# distance of vector to matrix rows
# keep numpy arrays x, B    
def distance_to_mat_euclidean2(nparray_double[:] x, 
                   nparray_double[:,:] B, double[:] res, 
                   long int n_B, long int p_B):
    
    cdef long int i

    for i in range(n_B):

        res[i] = euclidean_distance_c(x, B[i, :], p_B)
                            
    return res


# distance of vector to matrix rows
# keep numpy arrays x, B    
def distance_to_mat_manhattan2(nparray_double[:] x, 
                   nparray_double[:,:] B, double[:] res, 
                   long int n_B, long int p_B):
    
    cdef long int i

    for i in range(n_B):

        res[i] = manhattan_distance_c(x, B[i, :], p_B)
                            
    return res
        

# distance of vector to matrix rows
# keep numpy arrays x, B    
def distance_to_mat_cosine2(nparray_double[:] x, 
                   nparray_double[:,:] B, double[:] res, 
                   long int n_B, long int p_B):
    
    cdef long int i

    for i in range(n_B):

        res[i] = cosine_similarity_c(x, B[i, :], p_B)
                            
    return res

    
# find elt in list 
cdef long int find_elt_list(double elt, 
                            double[:] x, 
                            long int n_x):
    
    cdef long int j = 0
    
    for j in range(n_x):
        
        if x[j] == elt:
            
            return j        


# find_kmin_x
def find_kmin_x(x, n_x, k, cache=False):                 
    
    cdef str key 
    cdef double[:] sorted_x
    cdef dict res = {}
    cdef double elt = 0.0
    cdef int i = 0
    cdef long int idx 
    global __find_kmin_x_cache                          
    
    if cache: 
        
        key = str(hash(x)) 
        key += str(n_x) + str(k) 
    
        if key in __find_kmin_x_cache:
            
            return __find_kmin_x_cache[key]
        
        sorted_x = np.sort(x, kind='quicksort')        
    
        for i in range(k):
            
            elt = sorted_x[i]
            
            idx = find_elt_list(elt, x, n_x)
            
            res[idx] = elt
        
        __find_kmin_x_cache[key] = (np.asarray(list(res.values())), 
                                    np.asarray(list(res.keys()), 
                                    dtype=np.integer))                            
    
    # if cache == False    
    sorted_x = np.sort(x, kind='quicksort')        
    
    for i in range(k):
        
        elt = sorted_x[i]
        
        idx = find_elt_list(elt, x, n_x)
        
        res[idx] = elt
    
    return (np.asarray(list(res.values())), 
            np.asarray(list(res.keys()), 
            dtype=np.integer))                    


# calculate probs test i
def calculate_probs(long int[:] index, 
                    double[:,:] probs_train):
    
    cdef int idx, j
    cdef long int elt 
    cdef int n_classes = probs_train.shape[1]
    cdef double[:, ::1] probs_out = np.zeros((len(index), n_classes), 
                                              dtype=np.double)    
    
    idx = 0
    
    for elt in index:
        
        for j in range(n_classes):
        
            probs_out[idx, j] = probs_train[elt, j]
        
        idx += 1
    
    return np.asarray(probs_out)


# average probs
def average_probs(double[:,:] probs, double[:] weights):
    
    cdef double sum_probs
    cdef long int i 
    cdef long int n_obs = probs.shape[0]
    cdef int n_classes = probs.shape[1]
    #cdef double[:] probs_out = np.zeros(n_classes, dtype=np.double)
    cdef double[:] probs_out = np.zeros(n_classes)
    
    for j in range(n_classes):
        sum_probs = 0
        for i in range(n_obs):        
            sum_probs += probs[i, j]*weights[i]
        probs_out[j] = sum_probs
    
    return np.asarray(probs_out)
    

# calculate weights 
def calculate_weights(double[:] weights):
    cdef double[:] weights_out = np.zeros(len(weights))
    weights_out = 1/np.maximum(weights, np.finfo(float).eps)
    weights_out /= np.sum(weights_out)
    return weights_out


# 1 - model fitting -----

def fit_adaopt(X, y, 
        int n_iterations,
        long int n_X, int p_X, 
        int n_classes,
        double learning_rate, 
        double reg_lambda, 
        double reg_alpha,
        double eta, 
        double gamma, 
        double tolerance,
        str backend = "cpu"):
    
    
    cdef double[:, :] Y
    cdef double[:, :] probs
    cdef double[:,:] scaled_X
    cdef double[:,:] beta
    cdef long int[:] preds    
    cdef double[:] w_prev
    cdef double[:] w_new
    cdef double[:] w_m
    cdef long int[:] misclass
    cdef long int[:] misclass_condition    
    cdef double err_m, alpha_m    
    cdef int m = 0    
    cdef double v_prev = 0 
    cdef double tolerance_   
    cdef double[:] alphas = np.zeros(n_iterations)   
    cdef double err_bound = (1.0-1.0/n_classes)
    
    
    # obtain initial probs for each example
    scaled_X = X/norm(X, ord=2, axis=1)[:, None]                        
    Y = one_hot_encode(y, n_classes) 
    beta = lstsq(scaled_X, Y, rcond=None)[0]
    if backend != "cpu":
        device_put(np.asarray(scaled_X, dtype=np.float64))
        device_put(np.asarray(beta, dtype=np.float64))
    probs = expit(np.asarray(scaled_X, dtype=np.float64) @ np.asarray(beta, dtype=np.float64))
    probs /= np.sum(probs, axis=1)[:, None] 
    preds = np.asarray(probs).argmax(axis=1)    
    
    w_prev = np.repeat(1.0/n_X, n_X)    
    w_new = np.repeat(0.0, n_X)    
    w_m = np.repeat(1.0/n_X, n_X) 
    misclass = np.repeat(1000, n_X) 
    misclass_condition = np.repeat(0, n_X)  
    
    
    # loop update probs -----
    for m in range(n_iterations):
                
        # misclassification indicator for this iteration -----        
        for i in range(n_X):          
            if y[i] != preds[i]:                
                misclass[i] = preds[i]                
                misclass_condition[i] = 1  
            else:
                misclass[i] = 1000
                misclass_condition[i] = 0 
        
        
        # calculate error -----
        err_m = sum_product(misclass_condition, w_m, n_X) # sum(w_m) == 1  
        
        err_m += reg_lambda*(reg_alpha*cython_sum(np.abs(w_m), n_X) +\
                             (1 - reg_alpha)*0.5*cython_sum(np.power(w_m, 2), n_X))            
        err_m = min_c(max_c(err_m, 2.220446049250313e-16), 
                      err_bound)                
        
        
        # update weights based on misclassification error -----   
        alpha_m = learning_rate*log((n_classes - 1)*(1/err_m - 1))   
        alphas[m] = alpha_m
        w_prev = w_m     
        
        w_m *= np.exp(alpha_m*np.asarray(misclass_condition))
        w_m = np.asarray(w_m)/cython_sum(w_m, n_X)                                                    
        w_new = w_m
        
        # update probs based on misclassification error -----
        probs, v_prev = update_proba(probs, misclass,
                            w_new, w_prev, 
                            v_prev, eta, gamma, 
                            n_X, n_classes)        
        
        
        # recalculate preds -----
        preds = np.argmax(probs, axis=1) 
        
        
        # check tolerance -----               
        if  calculate_tolerance(np.asarray(alphas)) <= tolerance:
            n_iterations = m
            break               
    
    alphas_ = np.asarray(alphas)
    alphas_ = alphas_[np.where(alphas_ != 0)]
    probs_ = expit(np.asarray(probs))    
    probs_ /= np.sum(probs_, axis=1)[:, None]
        
    return {'probs': probs_, 
            'training_accuracy': np.mean(preds == np.asarray(y)),
            'alphas': alphas_, 
            'n_iterations': n_iterations, 
            'scaled_X_train': scaled_X}


# 2 - Model predict -----

def predict_proba_adaopt(X_test, 
                  scaled_X_train,
                  long int n_test, long int n_train,
                  double[:,::1] probs_train, int k,
                  int n_clusters, int seed,                   
                  int batch_size = 100,
                  type_dist="euclidean",    
                  cache=True, backend="cpu"):
    
    cdef int n_classes = probs_train.shape[1]        
    cdef int n_X_train = scaled_X_train.shape[0]
    cdef int p_X_train = scaled_X_train.shape[1]
    cdef double zero = np.finfo(float).eps
    cdef long int i, j
    cdef int m
    cdef tuple kmin_test_i
    cdef double[:] weights_test_i
    cdef long int[:] min_index_i, index_train
    
    cdef double[:,::1] dist_mat    
    cdef double[:,::1] out_probs = np.zeros((n_test, n_classes), 
                                            dtype=np.double)
    cdef double[:,::1] out_probs_ = np.zeros((n_test, n_classes), 
                                               dtype=np.double)
    cdef double[:,::1] scaled_X_test = X_test/norm(X_test, ord=2, axis=1)[:, None]
    cdef double[:,::1] probs_test_i = np.zeros((k, n_classes), 
                                               dtype=np.double)
    cdef double[:,::1] probs_train_ = np.zeros((n_train, n_classes), 
                                            dtype=np.double)
    cdef double[:] dists_test_i = np.zeros(n_train, dtype=np.double)
    cdef double[:] avg_probs_i = np.zeros(n_classes, dtype=np.double)
    cdef double[:] avg_probs = np.zeros(n_classes, dtype=np.double)
    
    
    # probabilities on training set -----
    
    if n_clusters <= 0: 

        # whole training set 
        probs_train_ = probs_train

    else: 

        # clustered training set 
        probs_train_ = np.zeros((n_clusters, n_classes), 
                                 dtype=np.double)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                 batch_size=batch_size, 
                                 random_state=seed).fit(scaled_X_train)

        scaled_X_train = kmeans.cluster_centers_                        

        for m in range(n_clusters):

            index_train = np.where(kmeans.labels_ == m)[0]

            avg_probs = np.average(calculate_probs(index_train, 
                                                   probs_train), 
                                   axis=0)

            for j in range(n_classes):

                probs_train_[m, j] = avg_probs[j]            


    # main loops -----

    if type_dist == "euclidean":

        for i in range(n_test):        

            dists_test_i = distance_to_mat_euclidean(scaled_X_test[i,:], 
                                           scaled_X_train)        

            kmin_test_i = find_kmin_x(dists_test_i, 
                                      n_x=n_train, 
                                      k=k, cache=cache) 

            weights_test_i = calculate_weights(kmin_test_i[0])

            probs_test_i = calculate_probs(kmin_test_i[1], 
                                           probs_train_)  

            avg_probs_i = average_probs(probs=probs_test_i, 
                                        weights=weights_test_i)

            for j in range(n_classes):

                out_probs[i, j] = avg_probs_i[j]

        out_probs_ = expit(out_probs)  

        out_probs_ /= np.sum(out_probs_, axis=1)[:, None]

        return np.asarray(out_probs_)


    if type_dist == "cosine":

        for i in range(n_test):        

            dists_test_i = distance_to_mat_cosine(scaled_X_test[i,:], 
                                           scaled_X_train)        

            kmin_test_i = find_kmin_x(dists_test_i, 
                                      n_x=n_train, 
                                      k=k, cache=cache) 

            weights_test_i = calculate_weights(kmin_test_i[0])

            probs_test_i = calculate_probs(kmin_test_i[1], 
                                           probs_train_)  

            avg_probs_i = average_probs(probs=probs_test_i, 
                                        weights=weights_test_i)

            for j in range(n_classes):

                out_probs[i, j] = avg_probs_i[j]

        out_probs_ = expit(out_probs)  

        out_probs_ /= np.sum(out_probs_, axis=1)[:, None]

        return np.asarray(out_probs_)


    if type_dist == "manhattan":

        for i in range(n_test):        

            dists_test_i = distance_to_mat_manhattan(scaled_X_test[i,:], 
                                           scaled_X_train)        

            kmin_test_i = find_kmin_x(dists_test_i, 
                                      n_x=n_train, 
                                      k=k, cache=cache) 

            weights_test_i = calculate_weights(kmin_test_i[0])

            probs_test_i = calculate_probs(kmin_test_i[1], 
                                           probs_train_)  

            avg_probs_i = average_probs(probs=probs_test_i, 
                                        weights=weights_test_i)

            for j in range(n_classes):

                out_probs[i, j] = avg_probs_i[j]

        out_probs_ = expit(out_probs)  

        out_probs_ /= np.sum(out_probs_, axis=1)[:, None]

        return np.asarray(out_probs_)


    if type_dist == "euclidean-f":                

        dist_mat = outer_sum_dot(np.asarray(scaled_X_test), 
                                 np.asarray(scaled_X_train), 
                                 backend)        

        for i in range(n_test):        

            dists_test_i = dist_mat[i, :]

            kmin_test_i = find_kmin_x(dists_test_i, 
                                      n_x=n_train, 
                                      k=k, cache=cache)    

            weights_test_i = calculate_weights(kmin_test_i[0])

            probs_test_i = calculate_probs(kmin_test_i[1], 
                                           probs_train_) 

            avg_probs_i = average_probs(probs=probs_test_i, 
                                        weights=weights_test_i)

            for j in range(n_classes):

                out_probs[i, j] = avg_probs_i[j]

        out_probs_ = expit(out_probs)    
        out_probs_ /= np.sum(out_probs_, axis=1)[:, None]

        del dist_mat
        gc.collect()

        return np.asarray(out_probs_)    
   
    
def predict_adaopt(double[:,::1] X_test, 
            double[:,::1] scaled_X_train,
            long int n_test, long int n_train,
            double[:,::1] probs_train, int k,
            int n_clusters, int seed,                   
            int batch_size = 100,
            type_dist="euclidean",
            cache=True):            
    
    return np.asarray(predict_proba_adaopt(X_test, scaled_X_train,
                                    n_test, n_train,
                                    probs_train, k, 
                                    n_clusters, seed,
                                    batch_size, type_dist,
                                    cache)).argmax(axis=1)
    