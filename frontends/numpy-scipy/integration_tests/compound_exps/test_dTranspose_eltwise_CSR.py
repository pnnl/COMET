import numpy as np
import scipy as sp
from cometpy import comet
import pytest

def run_numpy(A, B):
	C = A.transpose() * B

	return C

@comet.compile()
def run_comet_return(A, B):
	return A.transpose() * B

@comet.compile()
def run_comet_in_place(A, B, C):
	C[:] = A.transpose() * B

def init(data_rank2_path):
	B = sp.sparse.csr_array(sp.io.mmread(data_rank2_path))
	A = np.full([B.shape[1], B.shape[0]], 3.2,  dtype=float)
	Cnp = run_numpy(A, B)
	return A, B, Cnp

def test_dTranspose_eltwise_CSR_return(data_rank2_path):
	A, B, Cnp = init(data_rank2_path)
	Ccp = run_comet_return(A, B)
	if sp.sparse.issparse(Cnp):
		Cnp = Cnp.todense()
	if sp.sparse.issparse(Ccp):
		Ccp = Ccp.todense()
	np.testing.assert_almost_equal(Ccp, Cnp)

@pytest.mark.skip('In place mutation of sparse matrices is not supported yet')
def test_dTranspose_eltwise_CSR_in_place(data_rank2_path):
	A, B, Cnp = init(data_rank2_path)
	Ccp = sp.sparse.csr_array((B.shape[0], B.shape[1]), dtype = float) 
	run_comet_in_place(A, B, Ccp)
	if sp.sparse.issparse(Cnp):
		Cnp = Cnp.todense()
	if sp.sparse.issparse(Ccp):
		Ccp = Ccp.todense()
	np.testing.assert_almost_equal(Ccp, Cnp)
