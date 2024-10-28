import numpy as np
import pandas as pd


def create_histogram_with_bin_values(x):
    """
    Computes a histogram for the input data and assigns a value to each bin
    reflecting the ordering of the input.
    
    Args:
    x (list or np.array): Input data.
    num_bins (int): Number of bins for the histogram.
    
    Returns:
    bin_edges (np.array): The edges of the bins.
    bin_value_dict (dict): A dictionary where keys are the bin ranges (tuples) and values reflect the ordering.
    """
    # Compute the histogram
    _, bin_edges = np.histogram(x, bins="auto")

    bin_edges = np.concatenate([[-1e10], bin_edges, [1e10]]).ravel()        
        
    return {i: (bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)}


def assign_values_to_input(x, bin_value_dict):
    """
    Assigns values to a new input based on the provided bin ranges and values.

    Args:
    x (list or np.array): New input data to assign values to.
    bin_value_dict (dict): Dictionary where keys are bin ranges (tuples) and values are the assigned values.

    Returns:
    assigned_values (list): List of assigned values for the new input data.
    """

    if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.object_):

        return x

    assigned_values = []

    for value in x:
        assigned = None
        # Find the appropriate bin for each value
        for i, elt in enumerate(bin_value_dict.items()):
            if elt[1][0] < value <= elt[1][1]:
                assigned = float(i)
                break

        assigned_values.append(assigned)

    return np.asarray(assigned_values).ravel()


def get_histo_features(X, bin_value_dict=None):
    """
    Computes histogram features for the input data.
    
    Args:
    X {np.array or pd.DataFrame}: Input data.
    
    Returns:
    X_hist {np.array or pd.DataFrame}: Input data with histogram features.
    """

    if bin_value_dict is None: # training set case        

        if isinstance(X, pd.DataFrame):
            colnames = X.columns
            X = X.values
            X_hist = pd.DataFrame(np.zeros(X.shape), 
                                        columns=colnames)
            for i, col in enumerate(colnames):
                bin_value_dict = create_histogram_with_bin_values(X[:, i])
                X_hist[col] = assign_values_to_input(X[:, i], bin_value_dict)
        else: 
            X_hist = np.zeros(X.shape)        
            for i in range(X.shape[1]):
                bin_value_dict = create_histogram_with_bin_values(X[:, i])
                X_hist[:, i] = assign_values_to_input(X[:, i], bin_value_dict)            
            
        return X_hist, bin_value_dict

    else: # test set case
            
            if isinstance(X, pd.DataFrame):
                colnames = X.columns
                X = X.values
                X_hist = pd.DataFrame(np.zeros(X.shape), 
                                            columns=colnames)
                for i, col in enumerate(colnames):
                    X_hist[col] = assign_values_to_input(X[:, i], bin_value_dict)
            else: 
                X_hist = np.zeros(X.shape)        
                for i in range(X.shape[1]):
                    X_hist[:, i] = assign_values_to_input(X[:, i], bin_value_dict)            
    
            return X_hist