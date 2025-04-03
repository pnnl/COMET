import time
import numpy as np
import scipy as sp
import cupy as cp
from cometpy import comet
import sys
def run_numpy(A, B):
	C = A @ B

	return C

def run_cupy(A, B, C):
	with cp.cuda.Device(0):
		C = A @ B

	return C

@comet.compile(flags=None, target='gpu') # Default block-size(y=8, x=32, r=32)
def run_cometpy_default(A,B):
	C = A @ B

	return C

@comet.compile(flags='--gpu-block-y-size=64 --gpu-block-r-size=32 --gpu-block-x-size=64', target='gpu') # good for larger matrices
def run_cometpy_large(A,B):
	C = A @ B

	return C

@comet.compile(flags='--gpu-block-y-size=32 --gpu-block-r-size=32 --gpu-block-x-size=32', target='gpu')# good for smaller < 640 matrices
def run_cometpy_small(A,B):
	C = A @ B

	return C

size = int(sys.argv[1])
# for size in range(128, 4096+1, 128):
# for size in range(128, 1024+1, 1):
A = np.full([size,size], 2.2,  dtype=np.float64)
B = np.full([size,size], 3.4,  dtype=np.float64)
C = np.full([size,size], 0.0,  dtype=np.float64)
max_runs = 10
expected_result = run_numpy(A,B)
A_cp = cp.asarray(A)
B_cp = cp.asarray(B)
C_cp = cp.asarray(C)
res_cupy = run_cupy(A_cp,B_cp, C_cp)
for _ in range(max_runs):
	res_cupy = run_cupy(A_cp, B_cp, C_cp)

res_cupy = res_cupy.get()
expected_result = run_numpy(A,B)


if size < 768:
# Runs 10 times and outputs average
	res_cometpy = run_cometpy_small(A,B)
else:
	res_cometpy = run_cometpy_large(A,B)

np.testing.assert_almost_equal(res_cometpy, expected_result)
