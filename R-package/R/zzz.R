# global reference to packages (will be initialized in .onLoad)
cython <- NULL
numpy <- NULL
scipy <- NULL
sklearn <- NULL
sqlalchemy <- NULL
tqdm <- NULL
ms <- NULL


install_miniconda_ <- function(silent = TRUE)
{
  try(reticulate::install_miniconda(), silent = TRUE)
}

uninstall_mlsauce <- function(foo = NULL) {
  python <- reticulate:::.globals$py_config$python
  packages <- "mlsauce"
  args <- c("pip", "uninstall", "--yes", packages)
  result <- system2(python, args)
  if (result != 0L) {
    pkglist <- paste(shQuote(packages), collapse = ", ")
    msg <- paste("Error removing package(s):", pkglist)
    stop(msg, call. = FALSE)
  }
  packages
}

install_packages <- function(pip = TRUE) {

  has_cython <- reticulate::py_module_available("cython")
  has_numpy <- reticulate::py_module_available("numpy")
  has_scipy <- reticulate::py_module_available("scipy")
  has_sklearn <- reticulate::py_module_available("sklearn")
  has_tqdm <- reticulate::py_module_available("tqdm")
  has_mlsauce <- reticulate::py_module_available("mlsauce")
  has_pymongo <- reticulate::py_module_available("pymongo")
  has_querier <- reticulate::py_module_available("querier")
  has_sqlalchemy <- reticulate::py_module_available("sqlalchemy")


  if (has_cython == FALSE)
    reticulate::py_install("cython", pip = pip)

  if (has_pymongo == FALSE)
    reticulate::py_install("pymongo", pip = pip)

  if (has_querier == FALSE)
    reticulate::py_install("querier", pip = pip)

  if (has_numpy == FALSE)
    reticulate::py_install("numpy", pip = pip)

  if (has_scipy == FALSE)
    reticulate::py_install("scipy", pip = pip)

  if (has_sqlalchemy == FALSE)
    reticulate::py_install("sqlalchemy", pip = pip)

  if (has_sklearn == FALSE)
    reticulate::py_install("sklearn", pip = pip)

  if (has_tqdm == FALSE)
    reticulate::py_install("tqdm", pip = pip)

  if (has_mlsauce == FALSE)
    reticulate::py_install("mlsauce", pip = pip,
                           pip_ignore_installed = TRUE)
    #reticulate::py_install("git+https://github.com/thierrymoudiki/mlsauce.git",
    #                       pip = pip, pip_ignore_installed = TRUE)
}


.onLoad <- function(libname, pkgname) {

  do.call("uninstall_mlsauce", list(foo=NULL))

  do.call("install_miniconda_", list(silent=TRUE))

  do.call("install_packages", list(pip=TRUE))

  # use superassignment to update global reference to packages
  cython <<- reticulate::import("cython", delay_load = TRUE)
  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  pymongo <<- reticulate::import("pymongo", delay_load = TRUE)
  querier <<- reticulate::import("querier", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  sqlalchemy <<- reticulate::import("sqlalchemy", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ms <<- reticulate::import("mlsauce", delay_load = TRUE)

}
