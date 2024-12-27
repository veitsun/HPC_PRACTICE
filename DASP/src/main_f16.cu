#include "dasp_f16.h"

/**
 * @brief 对比两个数组cusp_val , cuda_val
 * 中的元素,判断它们是否在误差范围内。它使用一个重排索引数组 new_order ,
 * 以重新排序 cusp_val 中的内容， 与 cuda_val 中的元素逐一比较
 *
 * @param[out]  cusp_val 存储基准值的数组
 * @param[out]  cuda_val 存储对比值的数组
 * @param[out]  new_order 用于重排 cusp_val的整数索引数组
 * @param[in]   length 数组长度
 *
 * @return 所有元素对比通过 ，返回0
 */
int verify_new(MAT_VAL_TYPE *cusp_val, MAT_VAL_TYPE *cuda_val, int *new_order,
               int length) {
  for (int i = 0; i < length; i++) {
    int cusp_idx = new_order[i];
    float temp_cusp_val = cusp_val[cusp_idx];
    float temp_cuda_val = cuda_val[i];
    if (fabs(temp_cusp_val - temp_cuda_val) > 1) {
      // 之前在 1080 ti 上跑的时候，出现过这个问题
      printf(
          "error in (%d), cusp(%4.2f), cuda(%4.2f),please check your code!\n",
          i, temp_cusp_val, temp_cuda_val);
      return -1;
    }
  }
  printf("Y(%d), compute succeed!\n", length);
  return 0;
}

/**
 * @brief 该函数使用 CUDA 的 cuSPARSE 库来执行稀疏矩阵-向量乘法（SpMV, Sparse
 * Matrix-Vector
 * Multiplication），并测量操作的性能（包括预处理时间、计算时间、GFLOPS
 * 和内存带宽）。
 *
 * @param[out]  cu_ValA
 * @param[out]  cu_RowPtrA
 * @param[out]  cu_ColIdxA
 * @param[out]  cu_ValX 输入向量 X
 * @param[out]  cu_ValY
 * @param[in]   rowA
 * @param[in]   colA
 * @param[in]   nnzA    稀疏矩阵 A 的非零元素个数
 * @param[in]   data_origin1
 * @param[in]   data_origin2
 * @param[out]  pre_time
 * @param[out]  cu_time
 * @param[out]  cu_gflops
 * @param[out]  cu_bandwidth1
 * @param[out]  cu_bandwidth2
 *
 * @return
 */
__host__ void cusparse_spmv_all(MAT_VAL_TYPE *cu_ValA, MAT_PTR_TYPE *cu_RowPtrA,
                                int *cu_ColIdxA, MAT_VAL_TYPE *cu_ValX,
                                MAT_VAL_TYPE *cu_ValY, int rowA, int colA,
                                MAT_PTR_TYPE nnzA, long long int data_origin1,
                                long long int data_origin2, double *pre_time,
                                double *cu_time, double *cu_gflops,
                                double *cu_bandwidth1, double *cu_bandwidth2) {
  struct timeval t1, t2;

  MAT_VAL_TYPE *dA_val, *dX, *dY;
  int *dA_cid;
  MAT_PTR_TYPE *dA_rpt;
  float alpha = 1.0, beta = 0.0;
  // 将输入稀疏矩阵和向量从主机内存复制到设备内存。

  cudaMalloc((void **)&dA_val, sizeof(MAT_VAL_TYPE) * nnzA);
  cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA);
  cudaMalloc((void **)&dA_rpt, sizeof(MAT_PTR_TYPE) * (rowA + 1));
  cudaMalloc((void **)&dX, sizeof(MAT_VAL_TYPE) * colA);
  cudaMalloc((void **)&dY, sizeof(MAT_VAL_TYPE) * rowA);

  cudaMemcpy(dA_val, cu_ValA, sizeof(MAT_VAL_TYPE) * nnzA,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
  cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(MAT_PTR_TYPE) * (rowA + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dX, cu_ValX, sizeof(MAT_VAL_TYPE) * colA, cudaMemcpyHostToDevice);
  // cudaMemset(dY, 0.0, sizeof(MAT_VAL_TYPE) * rowA);

  // cusparse 初始化
  // 使用 cuSPARSE 提供的 cusparseSpMV 函数完成乘法计算。
  // 重复执行以测量性能。

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = NULL;
  size_t bufferSize = 0;

  gettimeofday(&t1, NULL);
  cusparseCreate(&handle);
  cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
  cusparseCreateDnVec(&vecX, colA, dX, CUDA_R_16F);
  cusparseCreateDnVec(&vecY, rowA, dY, CUDA_R_16F);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);
  gettimeofday(&t2, NULL);
  *pre_time =
      (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
  // printf("cusparse preprocessing time: %8.4lf ms\n", *pre_time);

  for (int i = 0; i < 100; ++i) {
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                 &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
  }
  cudaDeviceSynchronize();

  gettimeofday(&t1, NULL);
  for (int i = 0; i < 1000; ++i) {
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                 &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  *cu_time =
      ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) /
      1000;
  *cu_gflops = (double)((long)nnzA * 2) / (*cu_time * 1e6);
  *cu_bandwidth1 = (double)data_origin1 / (*cu_time * 1e6);
  *cu_bandwidth2 = (double)data_origin2 / (*cu_time * 1e6);
  // printf("cusparse:%8.4lf ms, %8.4lf Gflop/s, %9.4lf GB/s, %9.4lf GB/s\n",
  // *cu_time, *cu_gflops, *cu_bandwidth1, *cu_bandwidth2);

  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(handle);

  cudaMemcpy(cu_ValY, dY, sizeof(MAT_VAL_TYPE) * rowA, cudaMemcpyDeviceToHost);

  cudaFree(dA_val);
  cudaFree(dA_cid);
  cudaFree(dA_rpt);
  cudaFree(dX);
  cudaFree(dY);
}

// 加载输入稀疏矩阵文件，
// 调用两种 SpMV 实现进行性能测试
// 保存结果 ， 将测试结果记录到 CSV 文件中
// 验证计算正确性（可选）： 如果用户设置了验证选项 isverify 为 1
// ，将验证两种实现结果的一致性
// 在程序结束的时候释放分配的内存
__host__ int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Run the code by './spmv_half matrix.mtx $isverify(0 or 1)'.\n");
    return 0;
  }

  // struct timeval t1, t2;
  int rowA, colA;
  MAT_PTR_TYPE nnzA;
  int isSymmetricA;
  MAT_VAL_TYPE *csrValA;
  int *csrColIdxA;
  MAT_PTR_TYPE *csrRowPtrA;

  char *filename;
  filename = argv[1];           // matrix.mtx 路径
  int isverify = atoi(argv[2]); // 0 or 1
  int NUM = 4;
  int block_longest = 256;
  double threshold = 0.75;

  // ===test/cop20k_A.mtx===
  printf("\n===%s===\n\n", filename);
  // read matrix infomation from mtx file
  mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA,
                &csrValA, filename);
  // rowA 矩阵A的行数
  // colA 矩阵A的列数
  // nnzA 存储矩阵的非零元素个数
  // csrRowPtrA 存储矩阵A的行指针
  // csrColIdxA 存储矩阵A的列索引
  // csrValA 存储矩阵A的非零元素值
  // isSymmetricA 矩阵A是否对称
  // filename 矩阵文件名
  // filename = argv[1];           // matrix.mtx 路径
  // int isverify = atoi(argv[2]); // 0 or 1
  // int NUM = 4;
  // int block_longest = 256;
  // double threshold = 0.75;
  MAT_VAL_TYPE *X_val =
      (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA); // 输入向量
  initVec(X_val, colA);   // colA 大小的数组 ,初始化为 1.0
  initVec(csrValA, nnzA); // nnzA 大小的数组,初始化为 1.0

  MAT_VAL_TYPE *dY_val =
      (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA); // 存储结果
  MAT_VAL_TYPE *Y_val =
      (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA); // 存储结果
  int *new_order = (int *)malloc(sizeof(int) * rowA);      // 重排索引数组

  double pre_time = 0, cu_time = 0, cu_gflops = 0, cu_bandwidth1 = 0,
         cu_bandwidth2 = 0;
  //
  long long int data_origin1 = (nnzA + colA + rowA) * sizeof(MAT_VAL_TYPE) +
                               nnzA * sizeof(int) +
                               (rowA + 1) * sizeof(MAT_PTR_TYPE);
  long long int data_origin2 = (nnzA + nnzA + rowA) * sizeof(MAT_VAL_TYPE) +
                               nnzA * sizeof(int) +
                               (rowA + 1) * sizeof(MAT_PTR_TYPE);
  // cuSPARSE （NVIDIA 提供的稀疏矩阵计算库）
  cusparse_spmv_all(csrValA, csrRowPtrA, csrColIdxA, X_val, dY_val, rowA, colA,
                    nnzA, data_origin1, data_origin2, &pre_time, &cu_time,
                    &cu_gflops, &cu_bandwidth1, &cu_bandwidth2);

  double dasp_pre_time = 0, dasp_spmv_time = 0, dasp_spmv_gflops = 0,
         dasp_spmv_bandwidth = 0;
  // 定制的半精度 SPMV 实现
  // rowA 矩阵A的行数
  // colA 矩阵A的列数
  // nnzA 存储矩阵的非零元素个数
  // csrRowPtrA 存储矩阵A的行指针
  // csrColIdxA 存储矩阵A的列索引
  // csrValA 存储矩阵A的非零元素值
  // isSymmetricA 矩阵A是否对称
  // filename 矩阵文件名
  // filename 矩阵文件名
  // filename = argv[1];           // matrix.mtx 路径
  // int isverify = atoi(argv[2]); // 0 or 1
  // int NUM = 4;
  // int block_longest = 256;
  // double threshold = 0.75;
  // double dasp_pre_time = 0, dasp_spmv_time = 0, dasp_spmv_gflops = 0,
  //      dasp_spmv_bandwidth = 0;
  spmv_all(filename, csrValA, csrRowPtrA, csrColIdxA, X_val, Y_val, new_order,
           rowA, colA, nnzA, NUM, threshold, block_longest, &dasp_pre_time,
           &dasp_spmv_time, &dasp_spmv_gflops, &dasp_spmv_bandwidth);

  printf("                  pre_time     exe_time       performance\n");
  printf("DASP(Half):    %8.4lf ms  %8.4lf ms  %8.4lf GFlop/s\n", dasp_pre_time,
         dasp_spmv_time, dasp_spmv_gflops);
  printf("cusparse:      %8.4lf ms  %8.4lf ms  %8.4lf GFlop/s\n", pre_time,
         cu_time, cu_gflops);

  // 将测试结果记录到 CSV 文件中
  FILE *fout;
  fout = fopen("data/a100_f16_record.csv", "a");
  fprintf(fout, "%s,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf\n", filename, rowA, colA,
          nnzA, dasp_pre_time, dasp_spmv_time, dasp_spmv_gflops, pre_time,
          cu_time, cu_gflops);
  fclose(fout);
  // 验证计算正确性 （可选）
  if (isverify == 1) {
    int result = verify_new(dY_val, Y_val, new_order, rowA);
  }

  printf("\n");
  // 释放资源
  free(X_val);
  free(Y_val);
  free(dY_val);
  free(csrColIdxA);
  free(csrRowPtrA);
  free(csrValA);
  free(new_order);

  return 0;
}