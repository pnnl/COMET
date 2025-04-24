from cometpy import comet
import numpy as np


def mixed_numpy(A, B, N):
    for i in range(1, N-1):
        B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1])
    for i in range(1, N-1):
        A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);

    return A * B

@comet.compile()
def mixed_comet(A, B, N):

    for i in range(1, N-1):
        B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1])
    for i in range(1, N-1):
        A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    
    return A * B

def test_mixed():
    Anp = np.full([8], 1.0, dtype=float)
    Bnp = np.full([8], 1.0, dtype=float)
    mixed_numpy(Anp, Bnp, Anp.shape[0])
    Acp = np.full([8], 1.0, dtype=float)
    Bcp = np.full([8], 1.0, dtype=float)
    mixed_comet(Acp, Bcp, Acp.shape[0])

    np.testing.assert_almost_equal(Acp, Anp)
    np.testing.assert_almost_equal(Bcp, Bnp)

