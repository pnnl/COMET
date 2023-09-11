import time
import numpy as np
import scipy as sp
import comet

def run_numpy(A):
	var = A.sum()

	return var

@comet.compile(flags=None)
def run_comet_with_jit(A):
	var = A.sum()

	return var

A = sp.sparse.csr_matrix(sp.io.mmread("../../../integration_test/data/test_rank2.mtx"))
expected_result = run_numpy(A)
result_with_jit = run_comet_with_jit(A)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)
