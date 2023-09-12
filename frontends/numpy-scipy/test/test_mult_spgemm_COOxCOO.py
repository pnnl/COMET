
import numpy as np
import scipy as scp
import comet

@comet.compile(flags=None)
def compute_einsum_comet(A,B):
    
    C = comet.einsum('ai,ib->ab',A,B)

    return C



def compute_einsum_numpy(A,B):
    
    C_expected =  np.einsum('ai,ib->ab',A,B)    

    return C_expected


A = np.ones([7,5], dtype=np.float64)
B = np.full((5,4), 3, dtype=np.float64)
C = np.full((2), 0, dtype=np.float64)
 
result = compute_einsum_comet(scp.sparse.coo_matrix(A), scp.sparse.coo_matrix(B))
t2_expected_result = compute_einsum_numpy(A,B)

# np.testing.assert_array_equal(result.todense(), t2_expected_result)
np.testing.assert_array_equal(result, t2_expected_result)