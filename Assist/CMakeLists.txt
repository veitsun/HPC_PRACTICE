file(GLOB HELLOlib_Assit_cpp ${PROJECT_SOURCE_DIR}/Assist/*.cpp)
file(GLOB HELLOlib_Assit_cu ${PROJECT_SOURCE_DIR}/Assist/*.cu)


set(HELLOlib_Assist ${HELLOlib_Assit_cpp} ${HELLOlib_Assit_cu})

find_path(INCLUDE_DIR Harem.h HINTS ${CMAKE_SOURCE_DIR}/include)


if(NOT INCLUDE_DIR) 
  message(FATAL_ERROR "Harem.h not found !")
endif()

include_directories(INCLUDE_DIR)


add_library(SUNWEI_Assit SHARED  ${HELLOlib_Assist})

target_include_directories(SUNWEI_Assit
    PUBLIC
        ${INCLUDE_DIR}
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)
# 指定目标的包含目录
# target_include_directories(SUNWEI_Assit
#     PUBLIC
#         ${CMAKE_CURRENT_SOURCE_DIR}/../include
#     PRIVATE
#         ${CUDA_INCLUDE_DIRS}  
# )

# 链接 CUDA 库和其他必要的库
target_link_libraries(SUNWEI_Assit
    PUBLIC 
        ${CUDA_LIBRARIES}
)
