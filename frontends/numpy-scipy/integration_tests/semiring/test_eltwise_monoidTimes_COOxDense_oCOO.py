import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,ij->ij', A,B,semiring='*')

    return C


A = sp.sparse.coo_array(sp.io.mmread("../../../integration_test/data/test_rank2.mtx"))
B = np.full((A.shape), 2.7)

res = run_comet(A,B)
expected = sp.sparse.coo_array(([2.7,3.78,5.4,6.75,8.1,11.07,10.8,14.04,13.5], ([0,0,1,1,2,3,3,4,4], [0,3,1,4,2,0,3,1,4])))
np.testing.assert_almost_equal(res.todense(), expected.todense())