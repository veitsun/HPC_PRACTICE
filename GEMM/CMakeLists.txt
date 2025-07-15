set(INCLUDE_DIR_GEMM ${CMAKE_SOURCE_DIR}/GEMM/src/include)
list(APPEND INCLUDE_DIR_GEMM ${CMAKE_SOURCE_DIR}/src/include)

add_subdirectory(src)


# 定义宏
macro(add_multiple_executables)
    foreach(executable_name ${ARGN})
        add_executable(${executable_name} ${executable_name}.cu)
        target_include_directories(${executable_name} PUBLIC ${INCLUDE_DIR_GEMM})
        target_link_libraries(${executable_name} PRIVATE LIB_SRC LIB_GEMM_SRC ${CUBLAS_LIBRARY})
    endforeach()
endmacro()

# 调用宏
add_multiple_executables(
  GEMM_cublas
  GEMM_native
  GEMM_coalesce
)