set(targets nbody)
add_executable(nbody nbody.cc)

foreach(target IN LISTS ${targets})
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${target} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  FIND_PACKAGE(glfw3 REQUIRED)
  TARGET_LINK_LIBRARIES(${target} glfw)

  FIND_PACKAGE(OpenGL REQUIRED)
  TARGET_LINK_LIBRARIES(${target} OpenGL::GL)

  FIND_PACKAGE(Threads REQUIRED)
  TARGET_LINK_LIBRARIES(${target} Threads::Threads)

  TARGET_LINK_LIBRARIES(${target} sim)
  #TARGET_LINK_LIBRARIES(${target} vis)

  TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE "${CMAKE_SOURCE_DIR}/include/src")
endforeach()
