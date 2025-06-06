# Collect all .cpp and .cu in the src directory
file(GLOB Hellolib_src_cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB Hellolib_src_cu ${PROJECT_SOURCE_DIR}/src/*.cu)

# Combine both lists into Hellolib_src
set(Hellolib_src ${Hellolib_src_cpp} ${Hellolib_src_cu})

# 添加头文件内容
list(APPEND INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/include)


add_library(SUNWEI_SRC SHARED  ${Hellolib_src})

# 指定目标的包含目录
target_include_directories(SUNWEI_SRC
    PUBLIC
        ${INCLUDE_DIR}
    PRIVATE
        ${CUDA_INCLUDE_DIRS}  
)

# 链接 CUDA 库和其他必要的库
target_link_libraries(SUNWEI_SRC
    PUBLIC 
        ${CUDA_LIBRARIES}
)

