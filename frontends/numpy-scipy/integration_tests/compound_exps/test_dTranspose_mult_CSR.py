import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A, B):
	C = A.transpose() @ B

	return C

@comet.compile()
def run_comet_return(A, B):
	return A.transpose() @ B

@comet.compile()
def run_comet_in_place(A, B, C):
	C[:] = A.transpose() @ B

def init(data_rank2_path):
	B = sp.sparse.csr_array(sp.io.mmread(data_rank2_path))
	A = np.full([B.shape[0], 4], 1.7,  dtype=float)
	Cnp = run_numpy(A, B)
	return A, B, Cnp



def test_dTranspose_mult_CSR_in_place(data_rank2_path):
	A, B, Cnp = init(data_rank2_path)
	Ccp = np.full([4, B.shape[1]], 0.0,  dtype=float)
	run_comet_in_place(A, B, Ccp)
	if sp.sparse.issparse(Cnp):
		Cnp = Cnp.todense()
		Ccp = Ccp.todense()
	np.testing.assert_almost_equal(Ccp, Cnp)

def test_dTranspose_mult_CSR_return(data_rank2_path):
	A, B, Cnp = init(data_rank2_path)
	Ccp = run_comet_return(A, B)
	if sp.sparse.issparse(Cnp):
		Cnp = Cnp.todense()
		Ccp = Ccp.todense()
	np.testing.assert_almost_equal(Ccp, Cnp)
