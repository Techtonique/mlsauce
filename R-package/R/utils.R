# Check if package is available -----
is_package_available <- function(pkg_name)
{
  return(pkg_name %in% rownames(installed.packages()))
}
