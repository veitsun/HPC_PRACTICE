# Collect all .cpp in the src directory
file(GLOB HELLOlib_src_cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HELLOlib_src_cu ${PROJECT_SOURCE_DIR}/src/*.cu)

# Combine both lists into HELLOlib_src
set(HELLOlib_src ${HELLOlib_src_cpp} ${HELLOlib_src_cu})
# 查找 CUDA 包
find_package(CUDA REQUIRED)

find_path(INCLUDE_DIR Harem.h HINTS ${CMAKE_SOURCE_DIR}/include)

if(NOT INCLUDE_DIR) 
  message(FATAL_ERROR "Harem.h not found !")
endif()

include_directories(${INCLUDE_DIR})


add_library(SUNWEI_src SHARED  ${HELLOlib_src})
# 指定目标的包含目录
target_include_directories(SUNWEI_src
    PUBLIC
        ${INCLUDE_DIR}
    PRIVATE
        ${CUDA_INCLUDE_DIRS}  
)
# target_include_directories(SUNWEI_src
#     PUBLIC
#         ${CMAKE_CURRENT_SOURCE_DIR}/../include
#     PRIVATE
#         ${CUDA_INCLUDE_DIRS}  
# )

# 链接 CUDA 库和其他必要的库
target_link_libraries(SUNWEI_src 
    PUBLIC 
        ${CUDA_LIBRARIES}
)

