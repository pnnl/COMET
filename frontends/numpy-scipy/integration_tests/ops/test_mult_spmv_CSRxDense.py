import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A,B):
	C = A @ B 

	return C

@comet.compile(flags=None)
def run_comet_with_jit(A,B):
	C = A @ B 

	return C

A = sp.sparse.csr_array(sp.io.mmread("../../../../integration_test/data/test_rank2.mtx"))
B = np.full([A.shape[1]], 1.7,  dtype=float)
C = np.full([A.shape[0]], 0.0,  dtype=float)
expected_result = run_numpy(A,B)
result_with_jit = run_comet_with_jit(A,B)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)
