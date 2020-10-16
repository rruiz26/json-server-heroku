# init.R
#
# This file tells heroku to start up the server with R and which packages to install


#list packages here
my_packages = c("base","grf", "policytree")

install_if_missing = function(p) {
  if (p %in% rownames(installed.packages()) == FALSE) {
    install.packages(p)
  }
}

invisible(sapply(my_packages, install_if_missing))