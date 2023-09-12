

import numpy as np
import comet

#Testing Tensor Contractions with 2D input/output
@comet.compile(flags="")
def compute_einsum_2D_comet(A,B,E):
  
    C = comet.einsum('ij,jk->ik',A,B)
    D = comet.einsum('jli,ik->jlk',E,C)    

    return D

def compute_einsum_2D_numpy(A,B,E):
  
    C = np.einsum('ij,jk->ik',A,B)
    D = np.einsum('jli,ik->jlk',E,C)    

    return D

A = np.full((2,3),1, dtype=float)
B = np.full((3,4), 3, dtype=float)
E = np.full((3,5,2), 3, dtype=float)

result = compute_einsum_2D_comet(A,B,E)
expected_result = compute_einsum_2D_numpy(A,B,E)

np.testing.assert_array_equal(result, expected_result)


