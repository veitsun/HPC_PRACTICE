# set(CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_COMPILER "nvcc")

# 定义宏
macro(add_multiple_executables)
    foreach(executable_name ${ARGN})
        add_executable(${executable_name} ${executable_name}.cu)
        target_include_directories(${executable_name} PUBLIC ${INCLUDE_DIR})
        target_link_libraries(${executable_name} PRIVATE SUNWEI_SRC ${CUBLAS_LIBRARY})
    endforeach()
endmacro()

# 调用宏
add_multiple_executables(
  
  cp_async_example
  MulMatrixByCublas
  MulMatrixNative
  MulMatrixOnDevice
  MulMatrixOnDeviceOptBySharedMem
  MulMatrixOnDeviceOptBySharedMemAndRegister
  test
)



