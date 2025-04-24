import numpy as np
from cometpy import comet

def run_numpy(A,B):
	C = A @ B.transpose()

	return C

@comet.compile()
def run_comet_return(A,B):
	return A @ B.transpose()

@comet.compile()
def run_comet_in_place(A,B,C):
	C[:] = A @ B.transpose()

def init():
	A = np.full([5, 5], 2.3,  dtype=float)
	B = np.full([5, 5], 3.2,  dtype=float)
	Cnp = run_numpy(A, B)
	return A, B, Cnp

def test_Dense_mult_dTranspose_return():
	A, B, Cnp = init()
	Ccp = run_comet_return(A,B)
	np.testing.assert_almost_equal(Ccp, Cnp)

def test_Dense_mult_dTranspose_in_place():
	A, B, Cnp = init()
	Ccp = np.full([5, 5], 0.0,  dtype=float)
	run_comet_in_place(A,B, Ccp)
	np.testing.assert_almost_equal(Ccp, Cnp)
