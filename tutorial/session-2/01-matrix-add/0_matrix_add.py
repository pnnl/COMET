import time
import numpy as np
import scipy as sp
import cupy as cp
from cometpy import comet
import sys

# Numpy Matrix Addition
def run_numpy(A, B, C):
	D = A + B + C

	return D

# CometPy Matrix Addition on CPU
@comet.compile(flags=None) 
def run_cometpy_cpu(A,B):
	C = A + B

	return C

# CometPy Matrix Addition on GPU
@comet.compile(flags=None, target='gpu')
def run_cometpy_gpu(A,B,C):
	D = A + B + C

	return D


size = int(sys.argv[1])
# for size in range(128, 4096+1, 128):
# for size in range(128, 1024+1, 1):
A = np.full([size,size], 2.2,  dtype=np.float64)
B = np.full([size,size], 3.4,  dtype=np.float64)
C = np.full([size,size], 3.4,  dtype=np.float64)

# 0. Run numpy 
expected_result = run_numpy(A,B,C)

# 1. Run cometpy on cpu
res_cometpy = run_cometpy_cpu(A,B)

# Make sure results are correct
np.testing.assert_almost_equal(res_cometpy, expected_result)


############ UNCOMMENT ############
# # 2. Run cometpy on gpu
# res_cometpy = run_cometpy_gpu(A,B,C)

# # Make sure results are correct
# np.testing.assert_almost_equal(res_cometpy, expected_result)
