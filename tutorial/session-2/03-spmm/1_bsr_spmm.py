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


# A = sp.sparse.bsr_array(sp.io.mmread("../../data/cant.mtx"), dtype=float)
# A = sp.sparse.bsr_array(sp.io.mmread("../../data/cop20k_A.mtx"), dtype=float)
# A = sp.sparse.bsr_array(sp.io.mmread("../../data/pdb1HYS.mtx"), dtype=float)
# A = sp.sparse.bsr_array(sp.io.mmread("../../data/scircuit.mtx"), dtype=float)
B = np.random.rand(A.shape[1], 256)
C = np.full([A.shape[0], B.shape[1]], 0, dtype=np.float64)
expected_result = scipy_spmm(A,B)
# A_cp = cpx.scipy.sparse.csr_matrix(A)
# B_cp = cp.asarray(B)
# res_cupy = cupy_spmm(A_cp,B_cp)

# for _ in range(max_runs):
# 	res_cupy = cupy_spmm(A_cp, B_cp)

# res_cupy = res_cupy.get()
comet_result = cometpy_spmm(A,B)
np.testing.assert_almost_equal(comet_result, expected_result,1)

# int run_bsr(int num_rows, int num_cols, int nnzb, int blocksize, int* row_ptrs, int* colIndices, real* vals, int num_b_cols, real* dense_b, real* dense_C) {

bsr_lib = ctypes.cdll.LoadLibrary('./bsr.so')
func = bsr_lib.__getattr__('run_bsr')
func.argtypes = [ctypes.c_int] + 4 * [ctypes.c_int] + 2 * [ctypes.POINTER(ctypes.c_int)] + [ctypes.POINTER(ctypes.c_double)] + [ctypes.c_int] + 2 * [ctypes.POINTER(ctypes.c_double)]
Areshaped = A.data.reshape(-1)
func(max_runs, A.shape[0]//A.blocksize[0], A.shape[1]//A.blocksize[1], A.indices.shape[0], A.blocksize[0], A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), Areshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), B.shape[1], B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))