file(GLOB_RECURSE ALL_SOURCE_FILES 
    RELATIVE ${CMAKE_SOURCE_DIR}
    src/*.cc src/*.cu include/*.h test/*.cc)

find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
  message("Python not found")
endif()

find_file(CPP_LINT_PY 
  NAMES cpplint.py 
  HINTS ${CMAKE_SOURCE_DIR}
  DOC "Google cpp style scan program.")
if(NOT CPP_LINT_PY)
  message ("cpplint.py not found")
endif()

get_filename_component(ROOT_DIR ${CMAKE_SOURCE_DIR} PATH)
add_custom_target(check
  COMMAND "${CMAKE_COMMAND}" 
    -E chdir "${CMAKE_SOURCE_DIR}"
    ${CMAKE_SOURCE_DIR}/cpplint.py --root=${ROOT_DIR}
    ${ALL_SOURCE_FILES}
  DEPENDS ${AlL_SOURCE_FILES}
  COMMENT "Linting source code based on google code style"
  VERBATIM)
