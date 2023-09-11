import time
import numpy as np
import scipy as sp
import comet

def run_numpy(A,B):
	C = A.transpose().multiply(B)

	return C

@comet.compile(flags=None)
def run_comet_with_jit(A,B):
	C = A.transpose() * B

	return C

A = sp.sparse.csr_matrix(sp.io.mmread("../../../integration_test/data/test_rank2.mtx"))
B = sp.sparse.csr_matrix(sp.io.mmread("../../../integration_test/data/test_rank2.mtx"))
expected_result = run_numpy(A,B)
result_with_jit = run_comet_with_jit(A,B)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)