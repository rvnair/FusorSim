set(targets nbody)

foreach(target ${targets})
  message(STATUS "Initialize ${target}")
  add_executable(${target} ${target}.cc)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
  set_property(TARGET ${target} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  FIND_PACKAGE(Threads REQUIRED)
  TARGET_LINK_LIBRARIES(${target} Threads::Threads)

  TARGET_LINK_LIBRARIES(${target} sim)
  #TARGET_LINK_LIBRARIES(${target} vis)

  TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CMAKE_SOURCE_DIR}/inc/src)
endforeach()
