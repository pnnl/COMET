from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('i,ij->j', A,B,semiring='+,*')

    return C


B = sp.sparse.coo_array(sp.io.mmread("../../../../integration_test/data/test_rank2.mtx"))
A = np.full((B.shape[0]), 1.7)

res = run_comet(A,B)
expected = np.array([8.67,12.24,5.1,9.18,12.75]).reshape((B.shape[1]))
np.testing.assert_almost_equal(res, expected)