include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES}) 

# Add include directories of boost and python
find_package(PythonInterp)
find_package(PythonLibs)
find_package(Boost COMPONENTS python)
include_directories(${Boost_INCLUDE_DIRS} 
                    ${PYTHON_INCLUDE_DIRS}
                    clustering/)

# Set source files
file(GLOB_RECURSE 
     CLUSTERING_INTERFACE_SOURCES RELATIVE 
     ${CMAKE_CURRENT_SOURCE_DIR} clustering/*.cc)
message(STATUS "Clustering interface source files: " 
        ${CLUSTERING_INTERFACE_SOURCES})

# Python binding
python_add_module(${PROJECT_NAME}4Py ${CLUSTERING_INTERFACE_SOURCES})
target_link_libraries(${PROJECT_NAME}4Py 
  ${PROJECT_NAME} 
  ${Boost_LIBRARIES} 
  ${PYTHON_LIBRARIES})
add_dependencies(${PROJECT_NAME}4Py ${PROJECT_NAME})
