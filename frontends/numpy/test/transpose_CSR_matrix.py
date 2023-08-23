from cometpy import comet
import numpy as np
import scipy as scp

#Testing Dense tensor transpose
@comet.compile("")
def test_transpose(B):
   
    C = comet.einsum('ij->ji',B) 

    return C

def test_transpose_numpy(B):
   
    C_exp = np.einsum('ij->ji',B) 

    return C_exp

B = np.full((5,3), 2 , dtype=float)
tr = test_transpose(scp.sparse.csr_matrix(B))
tr_np = test_transpose_numpy(B)
np.testing.assert_array_equal(tr.todense(),tr_np)
