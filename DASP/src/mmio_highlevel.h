#ifndef _MMIO_HIGHLEVEL_

#define _MMIO_HIGHLEVEL_

#include "common.h"
#include "mmio.h"

/**
 * @brief
 * 实现了一个简单的专属扫描函数。专属扫描是一种广泛用于并行计算的操作，它计算数组的前缀和，但不包括当前元素本身。eclusive
 * 函数将输入数组中的每个元素替换为该元素前所有元素的和
 *
 * @param[out]  input   指向要进行专属扫描的数组
 * @param[in]   length  数组的长度
 *
 */
void exclusive_scan(MAT_PTR_TYPE *input, int length) {
  if (length == 0 || length == 1)
    return;

  MAT_PTR_TYPE old_val, new_val;

  old_val = input[0];
  input[0] = 0;
  for (int i = 1; i < length; i++) {
    new_val = input[i];
    input[i] = old_val + input[i - 1];
    old_val = new_val;
  }
}

// read matrix infomation from mtx file

int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, char *filename)

{

  int m_tmp, n_tmp, nnz_tmp;

  int ret_code;

  MM_typecode matcode;

  FILE *f;

  int nnz_mtx_report;

  int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0,
      isComplex = 0;

  // load matrix
  // 打开文件
  if ((f = fopen(filename, "r")) == NULL)

    return -1;
  // 读取矩阵头文件
  if (mm_read_banner(f, &matcode) != 0)

  {

    printf("Could not process Matrix Market banner.\n");

    return -2;
  }
  // 确定矩阵类型
  if (mm_is_pattern(matcode)) {
    isPattern = 1; /*printf("type = Pattern\n");*/
  }

  if (mm_is_real(matcode)) {
    isReal = 1; /*printf("type = real\n");*/
  }

  if (mm_is_complex(matcode)) {
    isComplex = 1; /*printf("type = real\n");*/
  }

  if (mm_is_integer(matcode)) {
    isInteger = 1; /*printf("type = integer\n");*/
  }

  /* find out size of sparse matrix .... */
  // 读取矩阵尺寸和非零元素个数
  ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);

  if (ret_code != 0)

    return -4;
  // 检查矩阵是否对称
  if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))

  {

    isSymmetric_tmp = 1;
    // 之前是注释掉的， 测试使用
    printf("input matrix is symmetric = true\n");

  }

  else

  {
    // 之前是注释掉的， 测试使用
    printf("input matrix is symmetric = false\n");
  }
  // 分配内存
  int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));

  memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

  int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

  int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

  MAT_VAL_TYPE *csrVal_tmp =
      (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  // 读取矩阵数据
  for (int i = 0; i < nnz_mtx_report; i++)

  {

    int idxi, idxj;

    double fval, fval_im;

    int ival;

    int returnvalue;

    if (isReal)

    {

      returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);

    }

    else if (isComplex)

    {

      returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);

    }

    else if (isInteger)

    {

      returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);

      fval = ival;

    }

    else if (isPattern)

    {

      returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);

      fval = 1.0;
    }

    // adjust from 1-based to 0-based
    // 1 based 索引 转换为 0 based 索引

    idxi--;

    idxj--;

    csrRowPtr_counter[idxi]++;

    csrRowIdx_tmp[i] = idxi;

    csrColIdx_tmp[i] = idxj;

    csrVal_tmp[i] = fval;
  }

  if (f != stdin)

    fclose(f);
  // 处理对称矩阵
  if (isSymmetric_tmp)

  {

    for (int i = 0; i < nnz_mtx_report; i++)

    {

      if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

        csrRowPtr_counter[csrColIdx_tmp[i]]++;
    }
  }

  // exclusive scan for csrRowPtr_counter

  int old_val, new_val;

  old_val = csrRowPtr_counter[0];

  csrRowPtr_counter[0] = 0;

  for (int i = 1; i <= m_tmp; i++)

  {

    new_val = csrRowPtr_counter[i];

    csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];

    old_val = new_val;
  }

  nnz_tmp = csrRowPtr_counter[m_tmp];

  *m = m_tmp;

  *n = n_tmp;

  *nnz = nnz_tmp;

  *isSymmetric = isSymmetric_tmp;

  // free tmp space

  free(csrColIdx_tmp);

  free(csrVal_tmp);

  free(csrRowIdx_tmp);

  free(csrRowPtr_counter);

  return 0;
}

// read matrix infomation from mtx file

int mmio_data(int *csrRowPtr, int *csrColIdx, MAT_VAL_TYPE *csrVal,
              char *filename)

{

  int m_tmp, n_tmp, nnz_tmp;

  int ret_code;

  MM_typecode matcode;

  FILE *f;

  int nnz_mtx_report;

  int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0,
      isComplex = 0;

  // load matrix

  if ((f = fopen(filename, "r")) == NULL)

    return -1;

  if (mm_read_banner(f, &matcode) != 0)

  {

    printf("Could not process Matrix Market banner.\n");

    return -2;
  }

  if (mm_is_pattern(matcode)) {
    isPattern = 1; /*printf("type = Pattern\n");*/
  }

  if (mm_is_real(matcode)) {
    isReal = 1; /*printf("type = real\n");*/
  }

  if (mm_is_complex(matcode)) {
    isComplex = 1; /*printf("type = real\n");*/
  }

  if (mm_is_integer(matcode)) {
    isInteger = 1; /*printf("type = integer\n");*/
  }

  /* find out size of sparse matrix .... */

  ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);

  if (ret_code != 0)

    return -4;

  if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))

  {

    isSymmetric_tmp = 1;

    // printf("input matrix is symmetric = true\n");

  }

  else

  {

    // printf("input matrix is symmetric = false\n");
  }

  int *csrRowPtr_counter = (int *)malloc((m_tmp + 1) * sizeof(int));

  memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

  int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

  int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

  MAT_VAL_TYPE *csrVal_tmp =
      (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (int i = 0; i < nnz_mtx_report; i++)

  {

    int idxi, idxj;

    double fval, fval_im;

    int ival;

    int returnvalue;

    if (isReal)

    {

      returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);

    }

    else if (isComplex)

    {

      returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);

    }

    else if (isInteger)

    {

      returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);

      fval = ival;

    }

    else if (isPattern)

    {

      returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);

      fval = 1.0;
    }

    // adjust from 1-based to 0-based

    idxi--;

    idxj--;

    csrRowPtr_counter[idxi]++;

    csrRowIdx_tmp[i] = idxi;

    csrColIdx_tmp[i] = idxj;

    csrVal_tmp[i] = fval;
  }

  if (f != stdin)

    fclose(f);

  if (isSymmetric_tmp)

  {

    for (int i = 0; i < nnz_mtx_report; i++)

    {

      if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

        csrRowPtr_counter[csrColIdx_tmp[i]]++;
    }
  }

  // exclusive scan for csrRowPtr_counter

  int old_val, new_val;

  old_val = csrRowPtr_counter[0];

  csrRowPtr_counter[0] = 0;

  for (int i = 1; i <= m_tmp; i++)

  {

    new_val = csrRowPtr_counter[i];

    csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];

    old_val = new_val;
  }

  nnz_tmp = csrRowPtr_counter[m_tmp];

  memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp + 1) * sizeof(int));

  memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int));

  if (isSymmetric_tmp)

  {

    for (int i = 0; i < nnz_mtx_report; i++)

    {

      if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

      {

        int offset =
            csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

        csrColIdx[offset] = csrColIdx_tmp[i];

        csrVal[offset] = csrVal_tmp[i];

        csrRowPtr_counter[csrRowIdx_tmp[i]]++;

        offset =
            csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];

        csrColIdx[offset] = csrRowIdx_tmp[i];

        csrVal[offset] = csrVal_tmp[i];

        csrRowPtr_counter[csrColIdx_tmp[i]]++;

      }

      else

      {

        int offset =
            csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

        csrColIdx[offset] = csrColIdx_tmp[i];

        csrVal[offset] = csrVal_tmp[i];

        csrRowPtr_counter[csrRowIdx_tmp[i]]++;
      }
    }

  }

  else

  {

    for (int i = 0; i < nnz_mtx_report; i++)

    {

      int offset =
          csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

      csrColIdx[offset] = csrColIdx_tmp[i];

      csrVal[offset] = csrVal_tmp[i];

      csrRowPtr_counter[csrRowIdx_tmp[i]]++;
    }
  }

  // free tmp space

  free(csrColIdx_tmp);

  free(csrVal_tmp);

  free(csrRowIdx_tmp);

  free(csrRowPtr_counter);

  return 0;
}
// read matrix infomation from mtx file
/**
 * @brief 读取一个存储在 Matrix Market
 * 格式文件中的稀疏矩阵，并将其转换为压缩稀疏行 （Compressed Sparse Row,
 * CSR）格式
 *
 * @param[out]  m            存储矩阵的行数
 * @param[out]  n            存储矩阵的列数
 * @param[out]  nnz          存储矩阵的非零元素个数
 * @param[out]  isSymmetric  存储矩阵是否对称
 * @param[out]  csrRowPtr    用于存储 CSR 格式的行指针数组的地址
 * @param[out]  csrColIdx    用于存储 CSR 格式的列索引数组的地址
 * @param[out]  csrVal       用于存储 CSR 格式的值数组的地址
 * @param[out]  filename     要读取的 Matrix Market 文件的路径
 *
 * @return 0 成功
 */
int mmio_allinone(int *m, int *n, MAT_PTR_TYPE *nnz, int *isSymmetric,
                  MAT_PTR_TYPE **csrRowPtr, int **csrColIdx,
                  MAT_VAL_TYPE **csrVal, char *filename) {
  int m_tmp, n_tmp;
  MAT_PTR_TYPE nnz_tmp;

  int ret_code;
  MM_typecode matcode;
  FILE *f;

  MAT_PTR_TYPE nnz_mtx_report;
  int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0,
      isComplex = 0;

  // load matrix
  if ((f = fopen(filename, "r")) == NULL)
    return -1;

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    return -2;
  }

  if (mm_is_pattern(matcode)) {
    isPattern = 1; /*printf("type = Pattern\n");*/
  }
  if (mm_is_real(matcode)) {
    isReal = 1; /*printf("type = real\n");*/
  }
  if (mm_is_complex(matcode)) {
    isComplex = 1; /*printf("type = real\n");*/
  }
  if (mm_is_integer(matcode)) {
    isInteger = 1; /*printf("type = integer\n");*/
  }

  /* find out size of sparse matrix .... */
  ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
  if (ret_code != 0)
    return -4;

  if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
    isSymmetric_tmp = 1;
    // printf("input matrix is symmetric = true\n");
  } else {
    // printf("input matrix is symmetric = false\n");
  }

  MAT_PTR_TYPE *csrRowPtr_counter =
      (MAT_PTR_TYPE *)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
  memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));

  int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
  int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
  MAT_VAL_TYPE *csrVal_tmp =
      (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
    int idxi, idxj;
    double fval, fval_im;
    int ival;
    int returnvalue;

    if (isReal) {
      returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
    } else if (isComplex) {
      returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
    } else if (isInteger) {
      returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
      fval = ival;
    } else if (isPattern) {
      returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
      fval = 1.0;
    }

    // adjust from 1-based to 0-based
    idxi--;
    idxj--;

    csrRowPtr_counter[idxi]++;
    csrRowIdx_tmp[i] = idxi;
    csrColIdx_tmp[i] = idxj;
    csrVal_tmp[i] = fval;
  }

  if (f != stdin)
    fclose(f);
  // 处理对称矩阵
  if (isSymmetric_tmp) {
    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
      if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
        csrRowPtr_counter[csrColIdx_tmp[i]]++;
    }
  }
  // 计算前缀和
  // exclusive scan for csrRowPtr_counter
  exclusive_scan(csrRowPtr_counter, m_tmp + 1);
  // 分配内存并初始化
  MAT_PTR_TYPE *csrRowPtr_alias =
      (MAT_PTR_TYPE *)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
  nnz_tmp = csrRowPtr_counter[m_tmp];
  int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
  MAT_VAL_TYPE *csrVal_alias =
      (MAT_VAL_TYPE *)malloc(nnz_tmp * sizeof(MAT_VAL_TYPE));

  memcpy(csrRowPtr_alias, csrRowPtr_counter,
         (m_tmp + 1) * sizeof(MAT_PTR_TYPE));
  memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));
  // 建立 CSR 格式矩阵
  if (isSymmetric_tmp) {
    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
      if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
        MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] +
                              csrRowPtr_counter[csrRowIdx_tmp[i]];
        csrColIdx_alias[offset] = csrColIdx_tmp[i];
        csrVal_alias[offset] = csrVal_tmp[i];
        csrRowPtr_counter[csrRowIdx_tmp[i]]++;

        offset = csrRowPtr_alias[csrColIdx_tmp[i]] +
                 csrRowPtr_counter[csrColIdx_tmp[i]];
        csrColIdx_alias[offset] = csrRowIdx_tmp[i];
        csrVal_alias[offset] = csrVal_tmp[i];
        csrRowPtr_counter[csrColIdx_tmp[i]]++;
      } else {
        MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] +
                              csrRowPtr_counter[csrRowIdx_tmp[i]];
        csrColIdx_alias[offset] = csrColIdx_tmp[i];
        csrVal_alias[offset] = csrVal_tmp[i];
        csrRowPtr_counter[csrRowIdx_tmp[i]]++;
      }
    }
  } else {
    for (MAT_PTR_TYPE i = 0; i < nnz_mtx_report; i++) {
      MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] +
                            csrRowPtr_counter[csrRowIdx_tmp[i]];
      csrColIdx_alias[offset] = csrColIdx_tmp[i];
      csrVal_alias[offset] = csrVal_tmp[i];
      csrRowPtr_counter[csrRowIdx_tmp[i]]++;
    }
  }
  // 传递结果
  *m = m_tmp;
  *n = n_tmp;
  *nnz = nnz_tmp;
  *isSymmetric = isSymmetric_tmp;

  *csrRowPtr = csrRowPtr_alias;
  *csrColIdx = csrColIdx_alias;
  *csrVal = csrVal_alias;
  // 释放内存
  // free tmp space
  free(csrColIdx_tmp);
  free(csrVal_tmp);
  free(csrRowIdx_tmp);
  free(csrRowPtr_counter);

  return 0;
}

#endif
