cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES "native")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(HPC_PRACTICE LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas=-v")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)



# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6/" CACHE PATH "CUDA Toolkit Root Directory")
# add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# 找cublas库
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARY NAMES cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 NO_DEFAULT_PATH)
if (NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "Could not find cuBLAS library")
else()
    message(STATUS "Found cuBLAS library: ${CUBLAS_LIBRARY}")
endif()

# 查找 OpenMP 包
# find_package(OpenMP REQUIRED)
# if (${OpenMP_FOUND})
#   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# else()
#   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
# endif()

set(CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_COMPILER "nvcc")

include_directories(${INCLUDE_DIR})
# include_directories(include)
# include_directories(kernels)
add_subdirectory(src)
add_subdirectory(Assist)
add_subdirectory(GEMM)
add_subdirectory(sunwei_test)
# add_subdirectory(kernels)


# 定义宏
macro(add_multiple_executables)
    foreach(executable_name ${ARGN})
        add_executable(${executable_name} ${executable_name}.cu)
        target_include_directories(${executable_name} PUBLIC ${INCLUDE_DIR})
        target_link_libraries(${executable_name} PRIVATE SUNWEI_src SUNWEI_Assit ${CUBLAS_LIBRARY})
        # 设置nvcc编译器参数
        # target_compile_options(${executable_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>)

    endforeach()
endmacro()

# 调用宏
add_multiple_executables(
  main
  MulMatrixByCublas
  MulMatrixNative
  MulMatrixOnDevice
  MulMatrixOnDeviceOptBySharedMem
  MulMatrixOnDeviceOptBySharedMemAndRegister
  test
)



