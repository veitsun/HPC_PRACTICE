set(INCLUDE_DIR_GEMM ${CMAKE_SOURCE_DIR}/GEMM/include)
list(APPEND INCLUDE_DIR_GEMM ${CMAKE_SOURCE_DIR}/src/include)

# 定义宏
macro(add_multiple_executables)
    foreach(executable_name ${ARGN})
        add_executable(${executable_name} ${executable_name}.cu)
        target_include_directories(${executable_name} PUBLIC ${INCLUDE_DIR_GEMM})
        target_link_libraries(${executable_name} PRIVATE SUNWEI_SRC ${CUBLAS_LIBRARY})
    endforeach()
endmacro()

# 调用宏
add_multiple_executables(
  helloworld
  GEMM_cublas
  GEMM_native
  GEMM_coalesce
)