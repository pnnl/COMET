from cometpy import comet
import numpy as np

def jacobi2D_numpy(A, B, N):
    for i in range(1, N-1):
        for j in range(1, N-1):
            B[i,j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j + 1] + A[i+1, j] + A[i-1, j])
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j + 1] + B[i+1, j] + B[i - 1, j]);

@comet.compile()
def jacobi2D_comet(A, B, N):
    #pragma parallel
    for i in range(1, N-1):
        #pragma parallel
        for j in range(1, N-1):
            B[i,j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j + 1] + A[i+1, j] + A[i-1, j])
    
    #pragma parallel
    for i in range(1, N-1):
        #pragma parallel
        for j in range(1, N-1):
            A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j + 1] + B[i+1, j] + B[i - 1, j]);

def test_jacobi2D():
    Anp = np.full([8, 8], 1.0, dtype=float)
    Bnp = np.full([8, 8], 1.0, dtype=float)
    jacobi2D_numpy(Anp, Bnp, Anp.shape[0])
    Acp = np.full([8, 8], 1.0, dtype=float)
    Bcp = np.full([8, 8], 1.0, dtype=float)
    jacobi2D_comet(Acp, Bcp, Acp.shape[0])
    
    np.testing.assert_almost_equal(Acp, Anp)
    np.testing.assert_almost_equal(Bcp, Bnp)