import numpy as np
from cometpy import comet

def run_numpy(A,B):
	return A.transpose() * B

@comet.compile(flags=None)
def run_comet_return(A,B,C):
	return A.transpose() * B

@comet.compile(flags=None)
def run_comet_in_place(A,B,C):
	C[:] = A.transpose() * B


def init():
	A = np.full([4, 4], 3.2,  dtype=float)
	B = np.full([4, 4], 2.0,  dtype=float)	
	Cnp = run_numpy(A, B)
	return A, B, Cnp

def test_dTranspose_eltwise_Dense_in_place():
	A, B, Cnp = init()
	Ccp = np.full([4, 4], 0.0,  dtype=float)
	run_comet_in_place(A, B, Ccp)
	np.testing.assert_almost_equal(Ccp, Cnp)

def test_dTranspose_eltwise_Dense_return():
	A, B, Cnp = init()
	Ccp = run_comet_return(A, B)
	np.testing.assert_almost_equal(Ccp, Cnp)
