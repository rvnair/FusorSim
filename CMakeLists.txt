cmake_minimum_required(VERSION 3.17.2)

project(Fusor-Simulation VERSION 0.1 LANGUAGES CXX)

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
      message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to
      remove CMakeCache.txt and CMakeFiles.")
endif()


ADD_SUBDIRECTORY(lib)
#ADD_SUBDIRECTORY(vis)
ADD_SUBDIRECTORY(src)
