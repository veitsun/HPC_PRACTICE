#include "myBase.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 16384
#define HEIGHT 16384
#define TILE_DIM 16
#define BLOCK_ROWS 16

// 用简单直接的全局内存方法进行图像旋转（用于验证结果）
__global__ void rotateImageGlobal(float *inputImage, float *outputImage,
                                  int width, int height, float angle) {
  // 计算线程对应的输出像素坐标
  int xOut = blockIdx.x * blockDim.x + threadIdx.x;
  int yOut = blockIdx.y * blockDim.y + threadIdx.y;

  if (xOut < width && yOut < height) {
    // 图像中心点
    float xCenter = width / 2.0f;
    float yCenter = height / 2.0f;

    // 将输出坐标转换为相对于中心的坐标
    float xRel = xOut - xCenter;
    float yRel = yOut - yCenter;

    // 应用逆旋转矩阵，计算对应的输入坐标
    float cosA = cosf(angle);
    float sinA = sinf(angle);
    float xIn = cosA * xRel + sinA * yRel + xCenter;
    float yIn = -sinA * xRel + cosA * yRel + yCenter;

    // 检查计算出的输入坐标是否在图像范围内
    if (xIn >= 0 && xIn < width && yIn >= 0 && yIn < height) {
      // 双线性插值
      int x0 = floorf(xIn);
      int y0 = floorf(yIn);
      int x1 = min(x0 + 1, width - 1);
      int y1 = min(y0 + 1, height - 1);

      float dx = xIn - x0;
      float dy = yIn - y0;

      float v00 = inputImage[y0 * width + x0];
      float v01 = inputImage[y0 * width + x1];
      float v10 = inputImage[y1 * width + x0];
      float v11 = inputImage[y1 * width + x1];

      float v0 = (1 - dx) * v00 + dx * v01;
      float v1 = (1 - dx) * v10 + dx * v11;

      outputImage[yOut * width + xOut] = (1 - dy) * v0 + dy * v1;
    } else {
      // 超出边界的点设为黑色
      outputImage[yOut * width + xOut] = 0.0f;
    }
  }
}

// 使用共享内存优化的图像旋转实现
__global__ void rotateImageSharedMem(float *inputImage, float *outputImage,
                                     int width, int height, float angle) {
  // 共享内存大小需要考虑旋转后需要访问的像素范围
  // 由于旋转，必须加载比TILE_DIM更大的区域
  extern __shared__ float sharedMem[];

  int blockX = blockIdx.x * TILE_DIM;
  int blockY = blockIdx.y * TILE_DIM;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 输出图像中的坐标
  int xOut = blockX + tx;
  int yOut = blockY + ty;

  // 图像中心
  float xCenter = width / 2.0f;
  float yCenter = height / 2.0f;

  // 旋转参数
  float cosA = cosf(angle);
  float sinA = sinf(angle);

  // 计算输出区块的边界框在输入图像中的映射范围
  float xCorners[4], yCorners[4];

  // 计算当前块四个角的坐标相对于中心的位置
  float x1Rel = blockX - xCenter; // 左上角
  float y1Rel = blockY - yCenter;
  float x2Rel = (blockX + TILE_DIM) - xCenter; // 右上角
  float y2Rel = blockY - yCenter;
  float x3Rel = blockX - xCenter; // 左下角
  float y3Rel = (blockY + TILE_DIM) - yCenter;
  float x4Rel = (blockX + TILE_DIM) - xCenter; // 右下角
  float y4Rel = (blockY + TILE_DIM) - yCenter;

  // 应用逆旋转变换到四个角
  xCorners[0] = cosA * x1Rel + sinA * y1Rel + xCenter;
  yCorners[0] = -sinA * x1Rel + cosA * y1Rel + yCenter;

  xCorners[1] = cosA * x2Rel + sinA * y2Rel + xCenter;
  yCorners[1] = -sinA * x2Rel + cosA * y2Rel + yCenter;

  xCorners[2] = cosA * x3Rel + sinA * y3Rel + xCenter;
  yCorners[2] = -sinA * x3Rel + cosA * y3Rel + yCenter;

  xCorners[3] = cosA * x4Rel + sinA * y4Rel + xCenter;
  yCorners[3] = -sinA * x4Rel + cosA * y4Rel + yCenter;

  // 确定需要加载的输入图像区域的边界
  float minX =
      fminf(fminf(xCorners[0], xCorners[1]), fminf(xCorners[2], xCorners[3]));
  float maxX =
      fmaxf(fmaxf(xCorners[0], xCorners[1]), fmaxf(xCorners[2], xCorners[3]));
  float minY =
      fminf(fminf(yCorners[0], yCorners[1]), fminf(yCorners[2], yCorners[3]));
  float maxY =
      fmaxf(fmaxf(yCorners[0], yCorners[1]), fmaxf(yCorners[2], yCorners[3]));

  // 向下取整并添加额外边距
  int startX = max(0, (int)floorf(minX) - 1);
  int startY = max(0, (int)floorf(minY) - 1);
  int endX = min(width - 1, (int)ceilf(maxX) + 1);
  int endY = min(height - 1, (int)ceilf(maxY) + 1);

  // 计算需要加载的区域大小
  int inputWidth = endX - startX + 1;
  int inputHeight = endY - startY + 1;

  // 协作加载输入数据到共享内存
  for (int y = ty; y < inputHeight; y += BLOCK_ROWS) {
    for (int x = tx; x < inputWidth; x += TILE_DIM) {
      int globalX = startX + x;
      int globalY = startY + y;

      if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
        sharedMem[y * inputWidth + x] = inputImage[globalY * width + globalX];
      } else {
        sharedMem[y * inputWidth + x] = 0.0f;
      }
    }
  }

  __syncthreads();

  // 处理当前线程的输出像素
  if (xOut < width && yOut < height) {
    // 计算相对于中心的坐标
    float xRel = xOut - xCenter;
    float yRel = yOut - yCenter;

    // 应用逆旋转矩阵
    float xIn = cosA * xRel + sinA * yRel + xCenter;
    float yIn = -sinA * xRel + cosA * yRel + yCenter;

    // 检查是否在图像边界内
    if (xIn >= 0 && xIn < width && yIn >= 0 && yIn < height) {
      // 转换到共享内存坐标系
      float xLocal = xIn - startX;
      float yLocal = yIn - startY;

      // 检查是否在加载的共享内存范围内
      if (xLocal >= 0 && xLocal < inputWidth - 1 && yLocal >= 0 &&
          yLocal < inputHeight - 1) {
        // 双线性插值
        int x0 = floorf(xLocal);
        int y0 = floorf(yLocal);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = xLocal - x0;
        float dy = yLocal - y0;

        float v00 = sharedMem[y0 * inputWidth + x0];
        float v01 = sharedMem[y0 * inputWidth + x1];
        float v10 = sharedMem[y1 * inputWidth + x0];
        float v11 = sharedMem[y1 * inputWidth + x1];

        float v0 = (1 - dx) * v00 + dx * v01;
        float v1 = (1 - dx) * v10 + dx * v11;

        outputImage[yOut * width + xOut] = (1 - dy) * v0 + dy * v1;
      } else {
        // 超出共享内存范围但在图像范围内
        // 在实际应用中应该从全局内存读取，但为简化代码，此处设为0
        outputImage[yOut * width + xOut] = 0.0f;
      }
    } else {
      // 超出图像边界
      outputImage[yOut * width + xOut] = 0.0f;
    }
  }
}

// 用于保存PGM格式图像
void savePGM(const char *filename, float *data, int width, int height) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "无法创建文件 %s\n", filename);
    return;
  }

  // 写入PGM头
  fprintf(fp, "P2\n%d %d\n255\n", width, height);

  // 写入像素数据
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int value = (int)(data[y * width + x] * 255.0f);
      value = value < 0 ? 0 : (value > 255 ? 255 : value);
      fprintf(fp, "%d ", value);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
  printf("已保存图像: %s\n", filename);
}

// 生成测试图像
void generateTestImage(float *image, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // 默认为黑色背景
      image[y * width + x] = 0.0f;

      // 绘制一个白色圆形
      float dx = x - width / 2.0f;
      float dy = y - height / 2.0f;
      float radius = width / 4.0f;
      if (dx * dx + dy * dy < radius * radius) {
        image[y * width + x] = 1.0f;
      }

      // 添加一个垂直线
      if (x == width / 2 && y > height / 2) {
        image[y * width + x] = 1.0f;
      }

      // 添加两个小垂直线段
      if ((x == width / 3 || x == 2 * width / 3) && y > 3 * height / 4 &&
          y < 7 * height / 8) {
        image[y * width + x] = 1.0f;
      }
    }
  }
}

int main() {
  float *h_input, *h_output, *d_input, *d_output;

  // 旋转角度 (30度，转换为弧度)
  float angle = 30.0f * M_PI / 180.0f;

  // 分配主机内存
  h_input = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
  h_output = (float *)malloc(WIDTH * HEIGHT * sizeof(float));

  // 生成测试图像
  generateTestImage(h_input, WIDTH, HEIGHT);

  // 保存原始图像
  savePGM("input.pgm", h_input, WIDTH, HEIGHT);

  // 分配设备内存
  cudaMalloc((void **)&d_input, WIDTH * HEIGHT * sizeof(float));
  cudaMalloc((void **)&d_output, WIDTH * HEIGHT * sizeof(float));

  // 将输入图像复制到设备
  cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float),
             cudaMemcpyHostToDevice);

  // 设置网格和块维度
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
  dim3 dimGrid((WIDTH + TILE_DIM - 1) / TILE_DIM,
               (HEIGHT + BLOCK_ROWS - 1) / BLOCK_ROWS);

  // 计算每个块需要的共享内存大小
  // 为简单起见，分配足够大的共享内存来处理各种旋转角度
  // 在实际应用中，可以根据旋转角度优化共享内存大小
  int padding = ceil(sqrt(2.0f) * std::max(TILE_DIM, BLOCK_ROWS)) + 2;
  int sharedMemSize =
      (TILE_DIM + 2 * padding) * (BLOCK_ROWS + 2 * padding) * sizeof(float);

  // 检查共享内存大小是否超过设备限制
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  if (sharedMemSize > deviceProp.sharedMemPerBlock) {
    fprintf(stderr, "警告: 共享内存请求 (%d bytes) 超过设备限制 (%d bytes)\n",
            sharedMemSize, deviceProp.sharedMemPerBlock);
    // 减小共享内存大小，实际应用中可能需要调整算法
    padding = floor((sqrt(deviceProp.sharedMemPerBlock / sizeof(float)) -
                     std::max(TILE_DIM, BLOCK_ROWS)) /
                    2);
    sharedMemSize =
        (TILE_DIM + 2 * padding) * (BLOCK_ROWS + 2 * padding) * sizeof(float);
  }

  // 使用全局内存版本进行旋转
  cudaMemset(d_output, 0, WIDTH * HEIGHT * sizeof(float));
  rotateImageGlobal<<<dimGrid, dimBlock>>>(d_input, d_output, WIDTH, HEIGHT,
                                           angle);
  cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float),
             cudaMemcpyDeviceToHost);
  savePGM("output_global.pgm", h_output, WIDTH, HEIGHT);

  // 使用共享内存版本进行旋转
  cudaMemset(d_output, 0, WIDTH * HEIGHT * sizeof(float));

  int repeat = 1;
  // 朴素的矩阵乘法

  startTimer();
  // for (int i = 0; i < repeat; i++) {

  rotateImageSharedMem<<<dimGrid, dimBlock, sharedMemSize>>>(
      d_input, d_output, WIDTH, HEIGHT, angle);
  // }
  // 执行纹理版本
  float time = stopTimer();

  printf("Time elapsed %f ms\n", time / repeat);

  cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float),
             cudaMemcpyDeviceToHost);
  savePGM("output_shared.pgm", h_output, WIDTH, HEIGHT);

  // 释放内存
  cudaFree(d_input);
  cudaFree(d_output);
  free(h_input);
  free(h_output);

  printf("旋转完成!\n");
  return 0;
}