find_file(DOXYGEN_CONFIG 
  NAMES doxygen_config 
  HINTS ${CMAKE_SOURCE_DIR}
  DOC "Doxygen configure file")

if(NOT DOXYGEN_CONFIG)
  message("Doxygen configure file not found")
endif()

find_package(Doxygen)
if(NOT DOXYGEN_FOUND)
  message("Doxygen not found(You might need to install the dot)")
else()
 add_custom_target(doc
    COMMAND "${CMAKE_COMMAND}" 
      -E chdir "${CMAKE_SOURCE_DIR}"
      doxygen ${DOXYGEN_CONFIG}
    DEPENDS ${DOXYGEN_CONFIG}
    COMMENT "Generate html based document"
    VERBATIM) 
endif()

