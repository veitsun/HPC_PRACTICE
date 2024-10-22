file(GLOB HELLOlib_kernels_cpp ${PROJECT_SOURCE_DIR}/kernels/*.cpp)
file(GLOB HELLOlib_kernels_cu ${PROJECT_SOURCE_DIR}/kernels/*.cu)
# file(GLOB HELLOlib_kernels_cuh ${PROJECT_SOURCE_DIR}/kernels/*.cuh)


# set(HELLOlib_kernels ${HELLOlib_kernels_cpp} ${HELLOlib_kernels_cu} ${HELLOlib_kernels_cuh})
set(HELLOlib_kernels ${HELLOlib_kernels_cpp} ${HELLOlib_kernels_cu})


# 查找 CUDA 包
find_package(CUDA REQUIRED)
add_library(SUNWEI_kernels SHARED  ${HELLOlib_kernels})

# 指定目标的包含目录
target_include_directories(SUNWEI_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        ${CUDA_INCLUDE_DIRS}  
)

# 链接 CUDA 库和其他必要的库
target_link_libraries(SUNWEI_kernels
    PUBLIC 
        ${CUDA_LIBRARIES}
)