# Collect all .cpp and .cu in the src directory
file(GLOB file_src_cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB file_src_cu ${PROJECT_SOURCE_DIR}/src/*.cu)

# Combine both lists into Hellolib_src
set(file_src ${file_src_cpp} ${file_src_cu})

# 添加头文件内容
# list(APPEND INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/include)
set(INCLUDE_GLOBAL_DIR ${CMAKE_SOURCE_DIR}/src/include)


add_library(LIB_SRC SHARED  ${file_src})

# 指定目标的包含目录
target_include_directories(LIB_SRC
    PUBLIC
        ${INCLUDE_GLOBAL_DIR}
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)

# 链接 CUDA 库和其他必要的库
target_link_libraries(LIB_SRC
    PUBLIC 
        ${CUDA_LIBRARIES}
)

