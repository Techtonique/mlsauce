# global reference to scipy (will be initialized in .onLoad)
ms <- NULL

.onLoad <- function(libname, pkgname) {

  if(is_package_available("reticulate") == FALSE)
  {
    install.packages("reticulate",
                     repos = c(CRAN="https://cloud.r-project.org"))
  }

  require("reticulate")

  try(reticulate::virtualenv_create('./r-reticulate',
                                    packages = c('numpy',
                                                 'scipy',
                                                 'Cython')),
      silent = TRUE)

  try(reticulate::use_virtualenv('./r-reticulate'),
      silent = TRUE)

  reticulate::virtualenv_install(packages = "git+https://github.com/Techtonique/mlsauce.git",
                                 envname = "./r-reticulate",
                                 pip_options = "--upgrade")

  # use superassignment to update global reference to package
  ms <<- reticulate::import("mlsauce", delay_load = TRUE)
}
