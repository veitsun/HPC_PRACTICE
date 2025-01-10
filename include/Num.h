#pragma once

#include <string>
using namespace std;
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
// const int n = 4092;
const int n = 5;

const int M = n;
const int N = n;
const int K = n;

const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 32;

// const int BLOCK_DIM_x = 16;
// const int BLOCK_DIM_y = 16;

const int elemNum = n * n;

const string INPUTFILENAME = "./data/input_data/random_5.txt";
