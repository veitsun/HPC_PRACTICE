# Collect all .cpp and .cu in the src directory
file(GLOB file_gemm_src_cpp ${PROJECT_SOURCE_DIR}/GEMM/src/*.cpp)
file(GLOB file_gemm_src_cu ${PROJECT_SOURCE_DIR}/GEMM/src/*.cu)

# Combine both lists into Hellolib_src
set(file_gemm_src ${file_gemm_src_cpp} ${file_gemm_src_cu})

# 添加头文件内容
set(INCLUDE_GEMM_DIR ${CMAKE_SOURCE_DIR}/gemm/src/include)


add_library(LIB_GEMM_SRC SHARED  ${file_gemm_src})

# 指定目标的包含目录
target_include_directories(LIB_GEMM_SRC
    PUBLIC
        ${INCLUDE_GLOBAL_DIR}
		${INCLUDE_GEMM_DIR}
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)

# 链接 CUDA 库和其他必要的库
target_link_libraries(LIB_GEMM_SRC
    PUBLIC 
        ${CUDA_LIBRARIES}
)

