#include(ExternalProject)

set(MKDIR mkdir kmlocal kmlocallib)
  execute_process(COMMAND ${MKDIR}
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
       )
message(STATUS "Command :  " ${MKDIR})

set(DOWNLOAD wget https://www.cs.umd.edu/~mount/Projects/KMeans/kmlocal-1.7.2.tar.gz)
  execute_process(COMMAND ${DOWNLOAD}
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kmlocal
       )
message(STATUS "Command :  " ${DOWNLOAD})

set(UNTAR tar -xvf kmlocal-1.7.2.tar.gz)
  execute_process(COMMAND ${UNTAR}
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kmlocal
       )
message(STATUS "Command :  " ${UNTAR})

set(KMLOCAL_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/kmlocal/kmlocal-1.7.2/src)

set(KMSOURCES ${KMLOCAL_INCLUDE_DIRS}/KCtree.cpp ${KMLOCAL_INCLUDE_DIRS}/KCutil.cpp ${KMLOCAL_INCLUDE_DIRS}/KM_ANN.cpp ${KMLOCAL_INCLUDE_DIRS}/KMcenters.cpp ${KMLOCAL_INCLUDE_DIRS}/KMdata.cpp ${KMLOCAL_INCLUDE_DIRS}/KMeans.cpp ${KMLOCAL_INCLUDE_DIRS}/KMfilterCenters.cpp ${KMLOCAL_INCLUDE_DIRS}/KMlocal.cpp ${KMLOCAL_INCLUDE_DIRS}/KMrand.cpp ${KMLOCAL_INCLUDE_DIRS}/KMterm.cpp)


set(BINARIES g++ -c ${KMSOURCES})

execute_process(COMMAND ${BINARIES}
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kmlocal/kmlocal-1.7.2/src
       )
message(STATUS "Command :  " ${BINARIES})

set(KMBINARIES ${KMLOCAL_INCLUDE_DIRS}/KCtree.o ${KMLOCAL_INCLUDE_DIRS}/KCutil.o ${KMLOCAL_INCLUDE_DIRS}/KM_ANN.o ${KMLOCAL_INCLUDE_DIRS}/KMcenters.o ${KMLOCAL_INCLUDE_DIRS}/KMdata.o ${KMLOCAL_INCLUDE_DIRS}/KMeans.o ${KMLOCAL_INCLUDE_DIRS}/KMfilterCenters.o ${KMLOCAL_INCLUDE_DIRS}/KMlocal.o ${KMLOCAL_INCLUDE_DIRS}/KMrand.o ${KMLOCAL_INCLUDE_DIRS}/KMterm.o)

set(ARCHIVE ar rvs libkmlocal.a ${KMBINARIES})

execute_process(COMMAND ${ARCHIVE}
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kmlocal/kmlocal-1.7.2/src
       )
message(STATUS "Command :  " ${ARCHIVE})

set(MVLIB mv libkmlocal.a ${CMAKE_CURRENT_BINARY_DIR}/kmlocallib)

execute_process(COMMAND ${MVLIB}
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kmlocal/kmlocal-1.7.2/src
       )
message(STATUS "Command :  " ${MVLIB})


