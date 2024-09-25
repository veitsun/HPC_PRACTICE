# Collect all .cpp in the src directory
file(GLOB HELLOlib_src_cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HELLOlib_src_cu ${PROJECT_SOURCE_DIR}/src/*.cu)

# Combine both lists into HELLOlib_src
set(HELLOlib_src ${HELLOlib_src_cpp} ${HELLOlib_src_cu})

# Create a static library from the collected source files
add_library(SUNWEI_src SHARED  ${HELLOlib_src})
