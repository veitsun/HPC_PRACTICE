cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(HPC_PRACTICE LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/include)

set(CUDA_ARCHITECTURES 86) # for RTX 3080ti
# set(CUDA_ARCHITECTURES 90) # for H100
set(CMAKE_CUDA_COMPILER "nvcc")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas=-v")
# 设置 pybind11 的包含路径
set(PYBIND11_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/pybind11/include)
list(APPEND INCLUDE_DIR ${PYBIND11_INCLUDE_DIR})

# 查找 CUDA 包
find_package(CUDA REQUIRED)
# 找cublas库
find_library(CUBLAS_LIBRARY NAMES cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 NO_DEFAULT_PATH)
if (NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "Could not find cuBLAS library")
else()
    message(STATUS "Found cuBLAS library: ${CUBLAS_LIBRARY}")
endif()

# 智能查找 Python：优先使用当前激活的环境
if(DEFINED ENV{CONDA_PREFIX})
    # 如果在 conda 环境中
    set(Python_ROOT_DIR $ENV{CONDA_PREFIX})
    message(STATUS "Detected conda environment: $ENV{CONDA_DEFAULT_ENV}")
elseif(DEFINED ENV{VIRTUAL_ENV})
    # 如果在 Python 虚拟环境中
    set(Python_ROOT_DIR $ENV{VIRTUAL_ENV})
    message(STATUS "Detected virtual environment: $ENV{VIRTUAL_ENV}")
endif()
# 明确查找 Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 查找 OpenMP 包
# find_package(OpenMP REQUIRED)
# if (${OpenMP_FOUND})
#   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# else()
#   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
# endif()

add_subdirectory(deps/pybind11)
add_subdirectory(src)
add_subdirectory(GEMM)
add_subdirectory(sunwei_test)


# 创建我的 python 模块
pybind11_add_module(my_module sunwei_test/my_module.cpp)
target_link_libraries(my_module PRIVATE LIB_SRC ${CUBLAS_LIBRARY})

# 定义宏
macro(add_multiple_executables)
    foreach(executable_name ${ARGN})
        add_executable(${executable_name} ${executable_name}.cu)
        target_include_directories(${executable_name} PUBLIC ${INCLUDE_DIR})
        target_link_libraries(${executable_name} PRIVATE LIB_SRC ${CUBLAS_LIBRARY})
    endforeach()
endmacro() 

# 调用宏
add_multiple_executables(
  gen_num
)



