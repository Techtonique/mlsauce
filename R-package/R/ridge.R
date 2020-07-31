


#' Ridge regressor
#'
#' @param reg_lambda L2 regularization parameter
#'
#' @return An object of class Ridge
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
#' obj <- mlsauce::RidgeRegressor()
#'
#' print(obj$get_params())
#'
#' obj$fit(X_train, y_train)
#'
#' print(obj$score(X_test, y_test))
#'
RidgeRegressor <- function(reg_lambda=0.1)
{

  ms$RidgeRegressor(reg_lambda=reg_lambda)
}
