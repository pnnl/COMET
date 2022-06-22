import numpy as np
from cometpy import comet

#Testing random tensor initialization
@comet.compile(flags="--convert-tc-to-ttgt")
def compute_randinit_comet(A,B):
   
    C = comet.einsum('ij,jk->ik',A,B)

    return C 

def compute_randinit_numpy(A,B):
   
    C_expected = np.einsum('ij,jk->ik',A,B)        

    return C_expected


A = np.array([[1.28, 1.50, 1.63, 1.79],[1.015, 1.37, 1.37, 1.31],[1.29, 1.97,  1.97, 1.99],[1.29, 1.78, 1.36, 1.24],[1.70,  1.05, 1.097, 1.183]],dtype=float) 
B = np.full((4,6), 3.0 , dtype=float)

elewise_result  = compute_randinit_comet(A,B)
elewise_expected_result = compute_randinit_numpy(A,B)

np.testing.assert_array_equal(elewise_result, elewise_expected_result)