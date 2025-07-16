#!/usr/bin/env bash
set -euo pipefail

# 可根据需要调整以下路径与参数
BUILD_DIR="build"
CMAKE_GENERATOR="Ninja"
CUDA_ROOT="/usr/local/cuda-12.6"
CXX_COMPILER="/usr/bin/clang++"
CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc"
CUDA_HOST_COMPILER="/usr/bin/g++"
CUDA_ARCH="86"

# 清理旧的构建产物
echo "清理旧构建目录：${BUILD_DIR}"
rm -rf "${BUILD_DIR:?}"/*

# 生成构建系统
echo "生成构建系统..."
cmake -S . -B "${BUILD_DIR}" \
  -G "${CMAKE_GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_ROOT}" \
  -DCMAKE_CXX_FLAGS="-I${CUDA_ROOT}/include" \
  -DCMAKE_CUDA_FLAGS="-I${CUDA_ROOT}/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${CUDA_ROOT}/lib64" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${CUDA_ROOT}/lib64 -lcudart"

# 执行构建
echo "开始构建..."
cmake --build "${BUILD_DIR}"
echo "构建完成。"