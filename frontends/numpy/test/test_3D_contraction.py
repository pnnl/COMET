
import numpy as np
from cometpy import comet

#Testing Tensor Contractions with 3D input/output
@comet.compile(flags=None)
def compute_einsum_3D_t1(A,B):
    C = comet.einsum('ijk,jil->kl',A,B) 

    return C

def compute_einsum_3D_t1_numpy(A,B):
    C_expected =  np.einsum('ijk,jil->kl',A,B)     

    return C_expected

@comet.compile(flags=None)
def compute_einsum_3D_t2(A,B):
    C = comet.einsum('iam,aj->ijm',A,B)  

    return C 

def compute_einsum_3D_t2_numpy(A,B):
   
    C_expected = np.einsum('iam,aj->ijm',A,B)      

    return C_expected


A = np.ones((3,4,5), dtype=float)
B = np.full((4,3,2), 3, dtype=float)

t1_result = compute_einsum_3D_t1(A,B) 
t1_expected_result = compute_einsum_3D_t1_numpy(A,B)

A = np.full((3,4,5), 1.2 ,dtype=float)
B = B.reshape(4,6)
t2_result = compute_einsum_3D_t2(A,B)
t2_expected_result = compute_einsum_3D_t2_numpy(A,B)

np.testing.assert_array_equal(t1_result, t1_expected_result)
np.testing.assert_almost_equal(t2_result, t2_expected_result)

