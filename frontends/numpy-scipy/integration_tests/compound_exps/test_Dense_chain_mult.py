from cometpy import comet
import numpy as np

@comet.compile(flags="")
def comet_chain_multiply_in_place(A,B,C,D,E):
    E[:] = A @ C @ B @ C.transpose() @ D

@comet.compile(flags="")
def comet_chain_multiply_return(A,B,C,D):
    return A @ C @ B @ C.transpose() @ D

def numpy_chain_multiply(A,B,C,D):
    return A @ C @ B @ C.transpose() @ D


def init():
    A = np.full([2,4], 1.0)
    B = np.full([4,4], 1.0)
    C = np.full([4,4], 1.0)
    D = np.full([4,6], 1.0)
    Enp = numpy_chain_multiply(A,B,C,D)

    return A, B, C, D, Enp

def test_Dense_chain_mult_in_place():
    A, B, C, D, Enp = init()
    Ecp = np.full([2,6], 0.0)
    comet_chain_multiply_in_place(A,B,C,D,Ecp)
    np.testing.assert_almost_equal(Ecp,Enp)

def test_Dense_chain_mult_return():
    A, B, C, D, Enp = init()
    Ecp = comet_chain_multiply_return(A,B,C,D)
    np.testing.assert_almost_equal(Ecp,Enp)