import time
import numpy as np
import scipy as sp
import cupy as cp
from cometpy import comet
import sys
def run_numpy(A, B):
	C = A @ B

	return C

def run_cupy(A, B):
	with cp.cuda.Device(0):
		C = A @ B

	return C

@comet.compile(flags='--gpu-block-y-size=32 --gpu-block-r-size=32 --gpu-block-x-size=32', target='gpu')
def run_cometpy(A,B):
	C = A @ B

	return C

size = int(sys.argv[1])
A = np.full([size,size], 2.2,  dtype=np.float64)
B = np.full([size,size], 3.4,  dtype=np.float64)
max_runs = 1
expected_result = run_numpy(A,B)
for _ in range(max_runs):
	A_cp = cp.array(A)
	B_cp = cp.array(B)
	res_cupy = run_cupy(A_cp,B_cp).get()
	A_cp = A_cp.get()
	B_cp = B_cp.get()
	res_cometpy = run_cometpy(A,B)

if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	res_cometpy = res_cometpy.todense()
	res_cupy = res_cupy.todense()
np.testing.assert_almost_equal(res_cometpy, expected_result)
np.testing.assert_almost_equal(res_cupy, expected_result)