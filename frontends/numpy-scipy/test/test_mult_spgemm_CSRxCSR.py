
import numpy as np
import scipy as scp
import comet

#Testing Tensor Contractions with 1D input/output
@comet.compile(flags=None)
def compute_einsum_1D_comet(A,B):
    
    C = comet.einsum('ai,ib->ab',A,B)

    return C



def compute_einsum_1D_numpy(A,B):
    
    C_expected =  np.einsum('ai,ib->ab',A,B)    

    return C_expected


A = np.ones([7,5], dtype=np.float64)
B = np.full((5,4), 3, dtype=np.float64)
C = np.full((2), 0, dtype=np.float64)

#Returns CSR 
result = compute_einsum_1D_comet(scp.sparse.csr_matrix(A), scp.sparse.csr_matrix(B))
t2_expected_result = compute_einsum_1D_numpy(A,B)

np.testing.assert_array_equal(result.todense(), t2_expected_result)