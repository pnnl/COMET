import time
import numpy as np
import scipy as sp
import cupy
from cometpy import comet
from conftest import gpu

def run_numpy(A,B,C):
	C[:] = A+B 

@comet.compile(flags=None, target="gpu")
def run_comet_with_jit(A,B,C):
	C[:] = A+B 

@gpu
def test_gpu_eltwise_add_dense_matrix_interop():
    A = np.full([4, 4], 2.2,  dtype=float)
    B = np.full([4, 4], 3.4,  dtype=float)
    C = np.full([4, 4], 0.0,  dtype=float)
    Ad = cupy.asarray(A)
    Bd = cupy.asarray(B)
    Cd = cupy.asarray(C)
    run_numpy(A,B,C)
    run_comet_with_jit(Ad,Bd,Cd)
    np.testing.assert_almost_equal(Cd.get(), C)
