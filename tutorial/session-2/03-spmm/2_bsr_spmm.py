import time
import numpy as np
import scipy as sp
from cometpy import comet
import cupy as cp
import cupyx as cpx
import ctypes
import sys

def scipy_spmm(A,B):
	C = A @ B

	return C

def cupy_spmm(A, B):
	with cp.cuda.Device(0):
		C = A @ B

	return C


max_runs = 10
matrix = sys.argv[1]
A = sp.sparse.bsr_array(sp.io.mmread(matrix), dtype=float)
bsize = A.blocksize[0]
if bsize < 4:
	bsize = 4
elif bsize < 8:
	bsize = 8
elif bsize < 16:
	bsize = 16

print(A.blocksize)
@comet.compile(flags='--gpu-block-y-size={0} --gpu-block-x-size=64 --gpu-block-r-size={0}'.format(bsize), target='gpu') # good for bsr
def cometpy_spmm(A,B):
	C = A @ B

	return C


B = np.random.rand(A.shape[1], 256)
C = np.full([A.shape[0], B.shape[1]], 0, dtype=np.float64)
expected_result = scipy_spmm(A,B)
A_cp = cpx.scipy.sparse.csr_matrix(A)
B_cp = cp.asarray(B)
res_cupy = cupy_spmm(A_cp,B_cp)

for _ in range(max_runs):
	res_cupy = cupy_spmm(A_cp, B_cp)

# res_cupy = res_cupy.get()
comet_result = cometpy_spmm(A,B)
np.testing.assert_almost_equal(comet_result, expected_result,1)