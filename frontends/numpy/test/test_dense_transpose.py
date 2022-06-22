
from cometpy import comet
import numpy as np

#Testing Dense tensor transpose
@comet.compile(flags="-opt-dense-transpose")
def dense_transpose_fill(B):
   
    C = comet.einsum('abij->baji',B) 

    return C

def dense_transpose_fill_numpy(B):
   
    C_exp = np.einsum('abij->baji',B) 

    return C_exp

B = np.full((6,8,8,6), 2 , dtype=float)
tr_fill = dense_transpose_fill(B)
tr_fill_exp = dense_transpose_fill_numpy(B)
np.testing.assert_array_equal(tr_fill,tr_fill_exp)
