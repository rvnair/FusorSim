cmake_minimum_required(VERSION 3.17.2)


project(Fusor-Simulation VERSION 0.1 LANGUAGES CXX)
find_package(CUDAToolkit)

set(CUDA_ARCH_RET 1)
if(CUDAToolkit_FOUND)
  message(STATUS "Figure out -arch")
  set(CUDA_ARCH_BINARYF ${CMAKE_BINARY_DIR}/detectgpu)
  set(CUDA_ARCH_CHECKER ${CMAKE_SOURCE_DIR}/scripts/detectgpu.cu)
  execute_process(COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${CUDA_ARCH_CHECKER} -o ${CUDA_ARCH_BINARYF})
  execute_process(COMMAND ${CUDA_ARCH_BINARYF} RESULT_VARIABLE CUDA_ARCH_RET OUTPUT_VARIABLE CUDA_ARC)
  if(${CUDA_ARCH_RET} EQUAL 0)
    set(CUDA_NVCC_FLAGS ${ARCH})
    project(Fusor-Simulation VERSION 0.1 LANGUAGES CXX CUDA)
  endif()
endif()

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
      message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to
      remove CMakeCache.txt and CMakeFiles.")
endif()


ADD_SUBDIRECTORY(lib)
#ADD_SUBDIRECTORY(vis)
ADD_SUBDIRECTORY(src)
