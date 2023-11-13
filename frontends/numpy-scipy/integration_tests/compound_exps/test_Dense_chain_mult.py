from cometpy import comet
import numpy as np

@comet.compile(flags="")
def comet_chain_multiply(A,B,C):
    D = comet.einsum('ij,jk,kl->il', A, C, B)

    return D

def numpy_chain_multiply(A,B,C):
    D = np.einsum('ij,jk,kl->il', A, C, B)


    return  D




A = np.full([2,4], 1.0)
B = np.full([4,4], 1.0)
C = np.full([4,4], 1.0)

Dn = numpy_chain_multiply(A,B,C)
Dc = comet_chain_multiply(A,B,C)

np.testing.assert_almost_equal(Dn,Dc)