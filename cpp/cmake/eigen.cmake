include(ExternalProject)

ExternalProject_Add(eigen
  URL http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/eigen
  INSTALL_COMMAND ""
)

# Specify include dir
ExternalProject_Get_Property(eigen source_dir)
set(EIGEN_INCLUDE_DIRS ${source_dir})
