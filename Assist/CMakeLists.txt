file(GLOB HELLOlib_Assit_cpp ${PROJECT_SOURCE_DIR}/Assist/*.cpp)
file(GLOB HELLOlib_Assit_cu ${PROJECT_SOURCE_DIR}/Assist/*.cu)


set(HELLOlib_Assist ${HELLOlib_Assit_cpp} ${HELLOlib_Assit_cu})


add_library(SUNWEI_Assit SHARED  ${HELLOlib_Assist})