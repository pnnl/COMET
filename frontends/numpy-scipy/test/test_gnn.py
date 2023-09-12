import scipy.sparse as sp
import scipy.io as io
import numpy as np
import comet

@comet.compile(flags="")
def run_pass_comet(A,W,X):
    Z = A @ X
    # for i in range(5):
    R = Z @ W

    return R

def run_pass_numpy(A,W,X):
    Z = A @ X
    # for i in range(5):
    R = Z @ W

    return R






G = io.mmread("../../integration_test/data/test_rank2.mtx")
A = sp.csr_matrix(G)
A = A + sp.csr_matrix(np.eye(A.shape[0], A.shape[1]))
X = np.ones((A.shape[0],5))
W = np.random.rand(X.shape[1],3)


comet_res = run_pass_comet(A,W,X)
numpy_res = run_pass_numpy(A,W,X)
np.testing.assert_array_almost_equal(comet_res,numpy_res)
