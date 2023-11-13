from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,jk->ik', A,B,semiring='+,*')

    return C


A = sp.sparse.csr_array(sp.io.mmread("../../../../integration_test/data/test_rank2.mtx"))
B = np.full((A.shape[1], 4), 1.7)

res = run_comet(A,B)
expected = np.array([4.08,4.08,4.08,4.08,7.65,7.65,7.65,7.65,5.1,5.1,5.1,5.1,13.77,13.77,13.77,13.77,17.34,17.34,17.34,17.34]).reshape((A.shape[0], 4))
np.testing.assert_almost_equal(res, expected)