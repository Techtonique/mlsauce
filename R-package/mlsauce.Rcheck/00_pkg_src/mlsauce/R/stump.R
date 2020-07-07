

# 1 - Classifier ----------------------------------------------------------

#' Stump classifier
#'
#' @param bins: int, number of histogram bins.
#'
#' @return An object of class StumpClassifier
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
#' X_train <- as.matrix(iris[train_index, 1:4])
#' y_train <- as.integer(iris[train_index, 5]) - 1L
#' X_test <- as.matrix(iris[test_index, 1:4])
#' y_test <- as.integer(iris[test_index, 5]) - 1L
#'
#' obj <- mlsauce::StumpClassifier()
#'
#' print(obj$get_params())
#'
#' obj$fit(X_train, y_train)
#'
#' print(obj$score(X_test, y_test))
#'
StumpClassifier <- function(bins="auto")
{
  ms$StumpClassifier(bins=bins)
}


# 2 - Regressor ----------------------------------------------------------


#' Stump Regressor
#'
