import time
import numpy as np
import scipy as sp
import comet

def run_numpy(B,C,D):
	T = B @ C
	A = T @ D
	return A

@comet.compile(flags=None)
def run_comet_with_jit(B,C,D):
	T = B @ C
	A = T @ D

	return A

B = sp.sparse.csr_matrix(sp.io.mmread("../../../integration_test/data/test_rank2.mtx"))
C = np.full([B.get_shape()[0], 4], 1.2,  dtype=float)
D = np.full([4, 4], 3.4,  dtype=float)
expected_result = run_numpy(B,C,D)
result_with_jit = run_comet_with_jit(B,C,D)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
if sp.sparse.issparse(result_with_jit):
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)
