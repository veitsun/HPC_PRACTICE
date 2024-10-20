
// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
const int N = 2048;
const int ThreadsPerBlock = 1024;
const int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

__global__ void BC_addKernel(const int *a, int *r) {
  __shared__ int cache
      [ThreadsPerBlock]; // 在编译时候就要确定，数组的大小,用于存储每个线程从全局内存中读取的数据
  int tid = blockIdx.x * blockDim.x +
            threadIdx.x; // 全局线程id， 用于确保每个线程在全局内存中的位置
  int cacheIndex = threadIdx.x; // 是线程在共享内存中的索引

  // copy data to shared memory from global memory
  cache[cacheIndex] = a
      [tid]; // 将数据从全局内存中拷贝到共享内存中，每个线程都会将全局内存中的数据复制到共享内存中
  __syncthreads(); // 同一个block内的线程同步，确保所有线程都完成数据复制后，再继续进行后续操作
  // add these data using reduce 数据是在共享内存中的
  for (
      int i = 1; i < blockDim.x;
      i *=
      2) { // 每次迭代中，线程都会将两个相邻的元素相加，并将结果存储到第一个元素位置
    // 这个循环的目的是逐步合并更多的数据，每次迭代合并的数据范围翻倍
    int index = threadIdx.x * 2 * i;
    // 确定了当前线程的起始位置。每次迭代的时候，i翻倍，使得合并的数据范围也翻倍。
    if (index < blockDim.x) {
      // 确保只有在index有效的情况下才进行加法操作，避免越界访问。
      cache[index] += cache[index + i];
    }
    __syncthreads();
    // 这里的同步为什么写在for循环的里面
    // 1，
    // 确保数据一致性：在每次迭代中，线程会读取和修改共享内存中的数据。如果不在每次迭代后进行同步，可能会导致数据竞争和不一致的问题。
    // 例如，在第一次迭代中，线程0和线程1分别更新cache[0] 和 cache[2],
    // 如果不同步，线程0可能会在线程1更新cache[2]
    // 之前就进入下一次迭代，导致错误结果。
    // 2，
    // 逐步规约：规约操作是逐步进行的，在每次的迭代中，线程会合并更大范围的数据。每次迭代后，所有线程必须同步，以确保前一次迭代的结果已经完全写入共享内存，才开始下一次迭代。
  }
  if (cacheIndex == 0) { // 将reduce结果写回到全局内存中
    r[blockIdx.x] = cache[cacheIndex];
  }
}

// 连续的规约求和
__global__ void NBC_addKernel2(const int *a, int *r) {
  __shared__ int cache[ThreadsPerBlock];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  // copy data to shared memory from global memory
  cache[cacheIndex] = a[tid];
  __syncthreads();

  // add these data using reduce
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (cacheIndex < i) {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
  }

  // copy the result of reduce to global memory
  if (cacheIndex == 0) {
    r[blockIdx.x] = cache[cacheIndex];
  }
}
// 由于每个线程的id与操作数据的编号一一对应，因此上述的代码很明显不会产生bank冲突。