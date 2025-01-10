# 找cublas库
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARY NAMES cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 NO_DEFAULT_PATH)
if (NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "Could not find cuBLAS library")
else()
    message(STATUS "Found cuBLAS library: ${CUBLAS_LIBRARY}")
endif()

set(CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_COMPILER "nvcc")

include_directories(${INCLUDE_DIR})

# 定义宏
macro(add_multiple_executables)
    foreach(executable_name ${ARGN})
        add_executable(${executable_name} ${executable_name}.cu)
        target_include_directories(${executable_name} PUBLIC ${INCLUDE_DIR})
        target_link_libraries(${executable_name} PRIVATE SUNWEI_src SUNWEI_Assit ${CUBLAS_LIBRARY})
    endforeach()
endmacro()

# 调用宏
add_multiple_executables(
  helloworld
  GEMM_cublas
  GEMM_native
)



