#include "myBase.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <math.h>
#include <stdio.h>

#define WIDTH 16384   // 图像宽度
#define HEIGHT 16384  // 图像高度
#define BLOCK_SIZE 16 // 线程块大小

//-------------------------------------------
// 简化版本的共享内存旋转函数 - 采用最基本的方法确保正确性
//-------------------------------------------
__global__ void rotateSharedMemory(float *input, float *output, int width,
                                   int height, float angle) {
  // 动态共享内存声明
  extern __shared__ float smem[];

  int tx = threadIdx.x, ty = threadIdx.y;
  int xOut = blockIdx.x * blockDim.x + tx;
  int yOut = blockIdx.y * blockDim.y + ty;

  // 提前返回超出图像边界的线程
  if (xOut >= width || yOut >= height)
    return;

  // 旋转中心
  float xCenter = width / 2.0f;
  float yCenter = height / 2.0f;

  // 计算sin和cos值
  float sinTheta = sin(angle);
  float cosTheta = cos(angle);

  // 计算需要加载的输入区域扩展大小
  int blockPad = ceil(sqrtf(2.0f) * BLOCK_SIZE) + 2; // 保证有足够的边界

  // 计算扩展区域的起始坐标
  int xStart = blockIdx.x * BLOCK_SIZE - blockPad;
  int yStart = blockIdx.y * BLOCK_SIZE - blockPad;

  // 共享内存的宽度和高度
  int smem_width = BLOCK_SIZE + 2 * blockPad;
  int smem_height = BLOCK_SIZE + 2 * blockPad;
  int smem_size = smem_width * smem_height;

  // 初始化共享内存为0 (协作式初始化)
  for (int i = ty * blockDim.x + tx; i < smem_size;
       i += blockDim.y * blockDim.x) {
    if (i < smem_size) {
      smem[i] = 0.0f;
    }
  }
  __syncthreads();

  // 将输入块加载到共享内存
  for (int y = yStart + ty; y < yStart + smem_height; y += blockDim.y) {
    for (int x = xStart + tx; x < xStart + smem_width; x += blockDim.x) {
      // 确保坐标在图像范围内
      if (x >= 0 && x < width && y >= 0 && y < height) {
        // 计算在共享内存中的索引
        int smem_idx = (y - yStart) * smem_width + (x - xStart);

        // 确保索引在共享内存范围内
        if (smem_idx >= 0 && smem_idx < smem_size) {
          smem[smem_idx] = input[y * width + x];
        }
      }
    }
  }

  // 确保所有线程完成共享内存的加载
  __syncthreads();

  // 计算输入坐标（以图像中心为旋转中心）
  // 注意：这里我们使用标准的旋转公式，从输出坐标反推输入坐标
  float xRelOut = xOut - xCenter;
  float yRelOut = yOut - yCenter;

  // 应用旋转变换（逆向映射）
  float xRelIn = xRelOut * cosTheta + yRelOut * sinTheta;
  float yRelIn = -xRelOut * sinTheta + yRelOut * cosTheta;

  // 计算输入坐标
  float xIn = xRelIn + xCenter;
  float yIn = yRelIn + yCenter;

  // 转换为共享内存的局部坐标
  float xLocal = xIn - xStart;
  float yLocal = yIn - yStart;

  // 边界检查，确保所需的插值点在共享内存范围内
  if (xLocal >= 1.0f && yLocal >= 1.0f && xLocal < (smem_width - 2) &&
      yLocal < (smem_height - 2)) {

    // 双线性插值
    int x0 = floor(xLocal);
    int y0 = floor(yLocal);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 计算插值权重
    float dx = xLocal - x0;
    float dy = yLocal - y0;

    // 读取四个相邻点
    float v00 = smem[y0 * smem_width + x0];
    float v01 = smem[y0 * smem_width + x1];
    float v10 = smem[y1 * smem_width + x0];
    float v11 = smem[y1 * smem_width + x1];

    // 双线性插值计算
    float v0 = v00 * (1.0f - dx) + v01 * dx;
    float v1 = v10 * (1.0f - dx) + v11 * dx;
    float value = v0 * (1.0f - dy) + v1 * dy;

    // 写入输出
    output[yOut * width + xOut] = value;
  } else {
    // 超出边界的像素设为黑色
    output[yOut * width + xOut] = 0.0f;
  }
}

//-------------------------------------------
// 直接在全局内存上实现旋转的简单版本（用于验证）
//-------------------------------------------
__global__ void rotateSimple(float *input, float *output, int width, int height,
                             float angle) {
  int xOut = blockIdx.x * blockDim.x + threadIdx.x;
  int yOut = blockIdx.y * blockDim.y + threadIdx.y;

  if (xOut >= width || yOut >= height)
    return;

  // 计算旋转参数
  float sinTheta = sin(angle);
  float cosTheta = cos(angle);
  float xCenter = width / 2.0f;
  float yCenter = height / 2.0f;

  // 计算输入坐标
  float xRelOut = xOut - xCenter;
  float yRelOut = yOut - yCenter;

  // 逆向映射
  float xRelIn = xRelOut * cosTheta + yRelOut * sinTheta;
  float yRelIn = -xRelOut * sinTheta + yRelOut * cosTheta;

  float xIn = xRelIn + xCenter;
  float yIn = yRelIn + yCenter;

  // 检查边界
  if (xIn >= 0 && xIn < width - 1 && yIn >= 0 && yIn < height - 1) {
    // 双线性插值
    int x0 = floor(xIn);
    int y0 = floor(yIn);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = xIn - x0;
    float dy = yIn - y0;

    float v00 = input[y0 * width + x0];
    float v01 = input[y0 * width + x1];
    float v10 = input[y1 * width + x0];
    float v11 = input[y1 * width + x1];

    float v0 = v00 * (1.0f - dx) + v01 * dx;
    float v1 = v10 * (1.0f - dx) + v11 * dx;
    float value = v0 * (1.0f - dy) + v1 * dy;

    output[yOut * width + xOut] = value;
  } else {
    output[yOut * width + xOut] = 0.0f;
  }
}

//-------------------------------------------
// 纹理内存优化版本
//-------------------------------------------
__global__ void rotateTextureMemory(float *output, int width, int height,
                                    float angle, cudaTextureObject_t texObj) {
  int xOut = blockIdx.x * blockDim.x + threadIdx.x;
  int yOut = blockIdx.y * blockDim.y + threadIdx.y;

  if (xOut >= width || yOut >= height)
    return;

  // 计算旋转参数
  float sinTheta = sin(angle);
  float cosTheta = cos(angle);
  float xCenter = width / 2.0f;
  float yCenter = height / 2.0f;

  // 计算输入坐标
  float xRelOut = xOut - xCenter;
  float yRelOut = yOut - yCenter;

  // 逆向映射
  float xRelIn = xRelOut * cosTheta + yRelOut * sinTheta;
  float yRelIn = -xRelOut * sinTheta + yRelOut * cosTheta;

  float xIn = xRelIn + xCenter;
  float yIn = yRelIn + yCenter;

  // 使用纹理进行采样
  output[yOut * width + xOut] = tex2D<float>(texObj, xIn, yIn);
}

//-------------------------------------------
// PGM文件保存函数
//-------------------------------------------
void savePGM(const char *filename, float *data, int width, int height) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    printf("无法创建文件 %s\n", filename);
    return;
  }

  // 写入PGM头
  fprintf(fp, "P2\n%d %d\n255\n", width, height);

  // 转换并写入像素数据
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int value = (int)(data[y * width + x] * 255);
      value = std::max(0, std::min(value, 255)); // 钳制到0-255范围
      fprintf(fp, "%d ", value);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
  printf("已保存图像: %s\n", filename);
}

//-------------------------------------------
// 主函数
//-------------------------------------------
int main() {
  float *h_input, *h_output, *d_input, *d_output;
  float angle = 30.0f * 3.14159265f / 180.0f; // 30度旋转角度（弧度）

  // 分配主机内存
  h_input = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
  h_output = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
  // 生成测试输入（从左到右渐变）
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      h_input[y * WIDTH + x] = x / (float)WIDTH; // 0.0~1.0渐变
    }
  }
  // 保存原始输入图像
  savePGM("original_input.pgm", h_input, WIDTH, HEIGHT);

  // 分配设备内存
  cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(float));
  cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float));
  cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float),
             cudaMemcpyHostToDevice);

  // 设置线程块和网格
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

  //-----------------------------------
  // 方案1：简单版本（用于验证）
  //-----------------------------------
  cudaMemset(d_output, 0, WIDTH * HEIGHT * sizeof(float));

  rotateSimple<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT, angle);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("简单版本内核执行错误: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("Simple Version: Rotation completed.\n");

  savePGM("simple_output.pgm", h_output, WIDTH, HEIGHT);

  //-----------------------------------
  // 方案2：共享内存版本执行
  //-----------------------------------

  // 计算扩展边界大小
  int blockPad = ceil(sqrtf(2.0f) * BLOCK_SIZE) + 2;
  int smem_width = BLOCK_SIZE + 2 * blockPad;
  int smem_height = BLOCK_SIZE + 2 * blockPad;

  // 计算所需共享内存大小
  int sharedMemSize = smem_width * smem_height * sizeof(float);

  // 检查共享内存大小是否超过设备限制
  int device;
  cudaGetDevice(&device);
  struct cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  if (sharedMemSize > deviceProp.sharedMemPerBlock) {
    printf("警告: 请求的共享内存大小 (%d 字节) 超过设备限制 (%d 字节)\n",
           sharedMemSize, deviceProp.sharedMemPerBlock);
    // 减小blockPad以适应内存限制
    blockPad = floor(
        (sqrt(deviceProp.sharedMemPerBlock / sizeof(float)) - BLOCK_SIZE) / 2);
    smem_width = BLOCK_SIZE + 2 * blockPad;
    smem_height = BLOCK_SIZE + 2 * blockPad;
    sharedMemSize = smem_width * smem_height * sizeof(float);
    printf("已调整: 新的blockPad=%d, 共享内存大小=%d\n", blockPad,
           sharedMemSize);
  }

  // 初始化输出为0
  cudaMemset(d_output, 0, WIDTH * HEIGHT * sizeof(float));

  // 使用共享内存版本进行旋转
  rotateSharedMemory<<<gridDim, blockDim, sharedMemSize>>>(
      d_input, d_output, WIDTH, HEIGHT, angle);

  // 检查错误
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("共享内存内核执行错误: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  // 拷贝结果回主机
  cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("Shared Memory Version: Rotation completed.\n");

  // 共享内存版本结果保存
  savePGM("shared_output.pgm", h_output, WIDTH, HEIGHT);

  //-----------------------------------
  // 方案3：纹理内存版本执行
  //-----------------------------------
  // 创建CUDA数组并绑定纹理
  cudaArray_t cuArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT);

  cudaMemcpy2DToArray(cuArray, 0, 0, h_input, WIDTH * sizeof(float),
                      WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice);

  // 创建纹理对象
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // 创建纹理对象描述符
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // 创建纹理对象
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // 初始化输出
  cudaMemset(d_output, 0, WIDTH * HEIGHT * sizeof(float));
  int repeat = 1;
  // 朴素的矩阵乘法

  startTimer();
  // for (int i = 0; i < repeat; i++) {

  rotateTextureMemory<<<gridDim, blockDim>>>(d_output, WIDTH, HEIGHT, angle,
                                             texObj);
  // }
  float time = stopTimer();

  printf("Time elapsed %f ms\n", time / repeat);
  // 检查错误
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("纹理内存内核执行错误: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  // 拷贝结果回主机
  cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("Texture Memory Version: Rotation completed.\n");

  // 纹理内存版本结果保存
  savePGM("texture_output.pgm", h_output, WIDTH, HEIGHT);

  //-----------------------------------
  // 释放资源
  //-----------------------------------
  cudaDestroyTextureObject(texObj); // 释放纹理对象
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeArray(cuArray);
  free(h_input);
  free(h_output);

  return 0;
}