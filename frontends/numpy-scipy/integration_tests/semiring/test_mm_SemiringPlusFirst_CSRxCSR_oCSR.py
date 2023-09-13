import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,jk->ik', A,B,semiring='+,first')

    return C


A = sp.sparse.csr_array(sp.io.mmread("../../../integration_test/data/test_rank2.mtx"))
B = sp.sparse.csr_array(sp.io.mmread("../../../integration_test/data/test_rank2.mtx"))

res = run_comet(A,B)
expected = sp.sparse.csr_array(([2.4,2.4,4.5,4.5,3,8.1,8.1,10.2,10.2], [0,3,1,4,2,0,3,1,4], [0,2,4,5,7,9]))
np.testing.assert_almost_equal(res.todense(), expected.todense())