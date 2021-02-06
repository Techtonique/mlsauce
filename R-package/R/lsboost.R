

# 1 - Classifier ----------------------------------------------------------

#' LSBoost classifier
#'
#' @param n_estimators: int, number of boosting iterations.
#' @param learning_rate: float, controls the learning speed at training time.
#' @param n_hidden_features: int
#' @param number of nodes in successive hidden layers.
#' @param reg_lambda: float, L2 regularization parameter for successive errors in the optimizer (at training time).
#' @param row_sample: float, percentage of rows chosen from the training set.
#' @param col_sample: float, percentage of columns chosen from the training set.
#' @param dropout: float, percentage of nodes dropped from the training set.
#' @param tolerance: float, controls early stopping in gradient descent (at training time).
#' @param direct_link: bool, indicates whether the original features are included (True) in model's fitting or not (False).
#' @param verbose: int, progress bar (yes = 1) or not (no = 0) (currently).
#' @param seed: int, reproducibility seed for nodes_sim=='uniform', clustering and dropout.
#' @param solver: str, type of 'weak' learner; currently in ('ridge', 'lasso')  
#' @param activation: str, activation function: currently 'relu', 'relu6', 'sigmoid', 'tanh'                 
#'
#' @return An object of class LSBoostClassifier
#' @export
#'
#' @examples
#'
#' library(datasets)
#'
#' X <- as.matrix(iris[, 1:4])
#' y <- as.integer(iris[, 5]) - 1L
#'
#' n <- dim(X)[1]
#' p <- dim(X)[2]
#' set.seed(21341)
#' train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
#' test_index <- -train_index
#' X_train <- as.matrix(X[train_index, ])
#' y_train <- as.integer(y[train_index])
#' X_test <- as.matrix(X[test_index, ])
#' y_test <- as.integer(y[test_index])
#'
#' obj <- mlsauce::LSBoostClassifier()
#'
#' print(obj$get_params())
#'
#' obj$fit(X_train, y_train)
#'
#' print(obj$score(X_test, y_test))
#'
LSBoostClassifier <- function(n_estimators=100L,
                              learning_rate=0.1,
                              n_hidden_features=5L,
                              reg_lambda=0.1,
                              row_sample=1,
                              col_sample=1,
                              dropout=0,
                              tolerance=1e-4,
                              direct_link=1L,
                              verbose=1L,
                              seed=123L, 
                              solver=c("ridge", "lasso"), 
                              activation="relu")
{

  ms$LSBoostClassifier(n_estimators=n_estimators,
                       learning_rate=learning_rate,
                       n_hidden_features=n_hidden_features,
                       reg_lambda=reg_lambda,
                       row_sample=row_sample,
                       col_sample=col_sample,
                       dropout=dropout,
                       tolerance=tolerance,
                       direct_link=direct_link,
                       verbose=verbose,
                       seed=seed, 
                       solver=match.arg(solver), 
                       activation=activation)
}


# 2 - Regressor ----------------------------------------------------------


#' LSBoost Regressor
#'
#' @param n_estimators: int, number of boosting iterations.
#' @param learning_rate: float, controls the learning speed at training time.
#' @param n_hidden_features: int
#' @param number of nodes in successive hidden layers.
#' @param reg_lambda: float, L2 regularization parameter for successive errors in the optimizer (at training time).
#' @param row_sample: float, percentage of rows chosen from the training set.
#' @param col_sample: float, percentage of columns chosen from the training set.
#' @param dropout: float, percentage of nodes dropped from the training set.
#' @param tolerance: float, controls early stopping in gradient descent (at training time).
#' @param direct_link: bool, indicates whether the original features are included (True) in model's fitting or not (False).
#' @param verbose: int, progress bar (yes = 1) or not (no = 0) (currently).
#' @param seed: int, reproducibility seed for nodes_sim=='uniform', clustering and dropout.
#' @param solver: str, type of 'weak' learner; currently in ('ridge', 'lasso')  
#' @param activation: str, activation function: currently 'relu', 'relu6', 'sigmoid', 'tanh'                 
#'
#' @return An object of class LSBoostRegressor
#' @export
#'
#' @examples
#'
#' library(datasets)
#'
#' X <- as.matrix(datasets::mtcars[, -1])
#' y <- as.integer(datasets::mtcars[, 1])
#'
#' n <- dim(X)[1]
#' p <- dim(X)[2]
#' set.seed(21341)
#' train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
#' test_index <- -train_index
#' X_train <- as.matrix(X[train_index, ])
#' y_train <- as.double(y[train_index])
#' X_test <- as.matrix(X[test_index, ])
#' y_test <- as.double(y[test_index])
#'
#' obj <- mlsauce::LSBoostRegressor()
#'
#' print(obj$get_params())
#'
#' obj$fit(X_train, y_train)
#'
#' print(obj$score(X_test, y_test))
#'
LSBoostRegressor <- function(n_estimators=100L,
                              learning_rate=0.1,
                              n_hidden_features=5L,
                              reg_lambda=0.1,
                              row_sample=1,
                              col_sample=1,
                              dropout=0,
                              tolerance=1e-4,
                              direct_link=1L,
                              verbose=1L,
                              seed=123L, 
                              solver=c("ridge", "lasso"),
                              activation="relu")
{

  ms$LSBoostRegressor(n_estimators=n_estimators,
                       learning_rate=learning_rate,
                       n_hidden_features=n_hidden_features,
                       reg_lambda=reg_lambda,
                       row_sample=row_sample,
                       col_sample=col_sample,
                       dropout=dropout,
                       tolerance=tolerance,
                       direct_link=direct_link,
                       verbose=verbose,
                       seed=seed, 
                       solver=match.arg(solver)
                       activation=activation)
}

