import time
import numpy as np
import scipy as sp
import comet

def run_numpy(A):
	B = np.einsum('ijk->jik', A)

	return B

@comet.compile(flags=None)
def run_comet_with_jit(A):
	B = comet.einsum('ijk->jik', A)

	return B

A = sp.sparse.coo_array(sp.io.mmread("../../../integration_test/data/test_rank3.tns"))
expected_result = run_numpy(A)
result_with_jit = run_comet_with_jit(A)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)
