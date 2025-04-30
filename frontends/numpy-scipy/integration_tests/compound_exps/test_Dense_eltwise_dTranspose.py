import numpy as np
from cometpy import comet

def run_numpy(A,B):
	return B * A.transpose()

@comet.compile()
def run_comet_return(A,B):
	return B * A.transpose()

@comet.compile()
def run_comet_in_place(A,B,C):
	C[:] = B * A.transpose()


def init():
	A = np.full([4, 4], 3.2,  dtype=float)
	B = np.full([4, 4], 2.0,  dtype=float)
	Cnp = run_numpy(A, B)
	return A, B, Cnp

def test_Dense_eltwise_dTranspose_in_place():
	A, B, Cnp = init()
	Ccp = np.full([4, 4], 0.0,  dtype=float)
	run_comet_in_place(A, B, Ccp)
	np.testing.assert_almost_equal(Cnp, Ccp)

def test_Dense_eltwise_dTranspose_return():
	A, B, Cnp = init()
	Ccp = run_comet_return(A, B)
	np.testing.assert_almost_equal(Cnp, Ccp)
