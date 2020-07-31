


#' Lasso regressor
#'
#' @param reg_lambda L1 regularization parameter
#' @param max_iter number of iterations of lasso shooting algorithm.
#' @param tol tolerance for convergence of lasso shooting algorithm.
#'
#' @return An object of class Lasso
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
#' obj <- mlsauce::Lasso()
#'
#' print(obj$get_params())
#'
#' obj$fit(X_train, y_train)
#'
#' print(obj$score(X_test, y_test))
#'
Lasso <- function(reg_lambda=0.1, max_iter=10L, tol=1e-3)
{

  ms$Lasso(reg_lambda=reg_lambda, max_iter=max_iter, tol=tol)
}
