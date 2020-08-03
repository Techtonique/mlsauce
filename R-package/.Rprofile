os <- reticulate::import("os")
PATH <- os$sys$executable

if (os$sys$version_info$major < 3)
{
 stop("Please upgrade to Python 3 \n Visit https://rstudio.github.io/reticulate/articles/versions.html")	
} else {
 Sys.setenv(RETICULATE_PYTHON = PATH)	
}