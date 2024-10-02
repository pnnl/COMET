import time
import numpy as np
import scipy as sp
import cupy as cp
from cometpy import comet
import sys
def run_numpy(A, B):
	C = A + B

	return C

def run_cupy(A, B, C):
	with cp.cuda.Device(0):
		C = A + B

	return C

@comet.compile(flags=None, target='gpu')
def run_cometpy(A,B):
	C = A + B

	return C

size = int(sys.argv[1])

A = np.full([size,size], 2.2,  dtype=np.float64)
B = np.full([size,size], 3.4,  dtype=np.float64)
C = np.full([size,size], 0.0,  dtype=np.float64)
max_runs = 10
expected_result = run_numpy(A,B)
A_cp = cp.asarray(A)
B_cp = cp.asarray(B)
C_cp = cp.asarray(C)
res_cupy = run_cupy(A_cp,B_cp, C_cp)

cp.cuda.runtime.deviceSynchronize()
for _ in range(max_runs):
	res_cupy = run_cupy(A_cp, B_cp, C_cp)
cp.cuda.runtime.deviceSynchronize()
res_cupy = res_cupy.get()

expected_result = run_numpy(A,B)
res_cometpy = run_cometpy(A,B)

np.testing.assert_almost_equal(res_cometpy, expected_result)
np.testing.assert_almost_equal(res_cupy, expected_result)