import time
import numpy as np
import scipy as sp
import comet

def run_numpy(L0,L1,L2):
	C = L0.multiply(L1 @ L2).sum() 
	return C

@comet.compile(flags=None)
def run_comet_with_jit(L0,L1,L2):
	C = comet.einsum('ij,jk->ik', L1,L2, mask=L0,mask_type="push")
	D = C.sum()
	
	return D

A = sp.sparse.csr_matrix(sp.io.mmread("../../../integration_test/data/tc.mtx"))
L0 = sp.sparse.tril(A, format='csr')
L1 = sp.sparse.tril(A, format='csr')
L2 = sp.sparse.tril(A, format='csr')
expected_result = run_numpy(L0,L1,L2)
result_with_jit = run_comet_with_jit(L0,L1,L2)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
if sp.sparse.issparse(result_with_jit):
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)
