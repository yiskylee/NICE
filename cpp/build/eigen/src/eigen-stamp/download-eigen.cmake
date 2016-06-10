message(STATUS "downloading...
     src='http://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2'
     dst='/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/3.2.8.tar.bz2'
     timeout='none'")




file(DOWNLOAD
  "http://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2"
  "/home/jason.b/Desktop/Github/NICE/cpp/build/eigen/src/3.2.8.tar.bz2"
  SHOW_PROGRESS
  # no EXPECTED_HASH
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'http://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
