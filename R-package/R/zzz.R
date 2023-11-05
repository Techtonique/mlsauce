# global reference to scipy (will be initialized in .onLoad)
ms <- NULL

.onLoad <- function(libname, pkgname) {
  try(reticulate::virtualenv_create('./r-reticulate'),
      silent = TRUE)
  try(reticulate::use_virtualenv('./r-reticulate'),
      silent = TRUE)
  # reticulate::py_install(
  #   "mlsauce",
  #   envname = "r-reticulate",
  #   pip = TRUE,
  #   pip_options = "--upgrade",
  #   pip_ignore_installed = TRUE
  # )
  reticulate::virtualenv_install(packages = "numpy",
                                 pip = TRUE,
                                 envname = "r-reticulate",
                                 pip_options = ">=1.13.0",
                                 pip_ignore_installed = TRUE)
  reticulate::virtualenv_install(packages = "scipy",
                                 pip = TRUE,
                                 envname = "r-reticulate",
                                 pip_options = ">=0.19.0",
                                 pip_ignore_installed = TRUE)
  reticulate::virtualenv_install(packages = "Cython",
                                 pip = TRUE,
                                 envname = "r-reticulate",
                                 pip_options = ">=0.29.21",
                                 pip_ignore_installed = TRUE)
  reticulate::virtualenv_install(packages = "git+https://github.com/Techtonique/mlsauce.git",
                                 envname = "r-reticulate",
                                 pip_options = "--upgrade")
  # use superassignment to update global reference to package
  ms <<- reticulate::import("mlsauce", delay_load = TRUE)
}
