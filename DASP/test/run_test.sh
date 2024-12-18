#!/bin/bash
cd ..
./spmv_half  test/cop20k_A.mtx 1
./spmv_double  test/cop20k_A.mtx 1

