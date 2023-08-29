

import numpy as np
import comet

#Testing Vector @ 2D Dense Matrix Multiply
@comet.compile(flags="")
def compute_einsum_2D_comet(A,B):
  
    C = A @ B

    return C

def compute_einsum_2D_numpy(A,B):
  
    C = A @ B

    return C

A = np.full((2),1, dtype=float)
B = np.full((2,3), 3, dtype=float)

result = compute_einsum_2D_comet(A,B)
expected_result = compute_einsum_2D_numpy(A,B)

np.testing.assert_array_equal(result, expected_result)


