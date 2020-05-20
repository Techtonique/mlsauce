


#' AdaOpt classifier
#'
#' @param n_iterations number of iterations of the optimizer at training time
#' @param learning_rate controls the speed of the optimizer at training time
#' @param reg_lambda L2 regularization parameter for successive errors in the optimizer (at training time)
#' @param reg_alpha L1 regularization parameter for successive errors in the optimizer (at training time)
#' @param eta controls the slope in gradient descent (at training time)
#' @param gamma controls the step size in gradient descent (at training time)
#' @param k number of nearest neighbors selected at test time for classification
#' @param tolerance controls early stopping in gradient descent (at training time)
#' @param n_clusters number of clusters, if MiniBatch k-means is used at test time (for faster prediction)
#' @param batch_size size of the batch, if MiniBatch k-means is used at test time (for faster prediction)
#' @param row_sample percentage of rows chosen from training set (by stratified subsampling, for faster prediction)
#' @param type_dist distance used for finding the nearest neighbors; currently `euclidean-f` (euclidean distances
#' calculated as whole), `euclidean` (euclidean distances calculated row by row), `cosine` (cosine distance)
#' @param cache if the nearest neighbors are cached or not, for faster retrieval in subsequent calls
#' @param seed reproducibility seed for initial weak learner and clustering
#'
#' @return An object of class AdaOpt
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
#' obj <- AdaOpt()
#'
#' obj$fit(X_train, y_train)
#'
#' print(obj$score(X_test, y_test))
#'
AdaOpt <- function(n_iterations=50L,
                  learning_rate=0.3,
                  reg_lambda=0.1,
                  reg_alpha=0.5,
                  eta=0.01,
                  gamma=0.01,
                  k=3L,
                  tolerance=0,
                  n_clusters=0,
                  batch_size=100L,
                  row_sample=1.0,
                  type_dist="euclidean-f",
                  cache=TRUE,
                  seed=123L)
{

  ms$AdaOpt(n_iterations=n_iterations,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            eta=eta,
            gamma=gamma,
            k=k,
            tolerance=tolerance,
            n_clusters=n_clusters,
            batch_size=batch_size,
            row_sample=row_sample,
            type_dist=type_dist,
            cache=cache,
            seed=seed)
}
