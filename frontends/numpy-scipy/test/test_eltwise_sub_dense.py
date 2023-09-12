import numpy as np
import comet

#Testing Tensor Subtraction with 1D input/output
@comet.compile(flags=None)
def compute_sub_1D_comet(A,B):
    
    C = comet.einsum('ai,iam->m',A,B)
    C = C - C 
    return C


def compute_sub_1D_numpy(A,B):
    
    C_expected =  np.einsum('ai,iam->m',A,B)    
    C_expected = C_expected - C_expected
    return C_expected

A = np.ones([4,5], dtype=float)
B = np.full((5,4,2), 3, dtype=float)
 
t2_result = compute_sub_1D_comet(A,B)
t2_expected_result = compute_sub_1D_numpy(A,B)

np.testing.assert_array_equal(t2_result, t2_expected_result)
