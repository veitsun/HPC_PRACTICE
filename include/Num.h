#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
const int n = 1024;

const int nx = n;
const int ny = n;

const int M = n;
const int N = n;
const int K = n;

const int BLOCK_DIM_x = 32;
const int BLOCK_DIM_y = 32;

// const int BLOCK_DIM_x = 16;
// const int BLOCK_DIM_y = 16;

const int elemNum = nx * ny;
