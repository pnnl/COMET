
import numpy as np
import comet

#Testing elementwise multiplication

@comet.compile(flags=None)
def compute_mult_transpose(a19_aaaa,t2_aaaa):
   
    t2_tr = comet.einsum('ij->ji',t2_aaaa)
    r2_aaaa  =  a19_aaaa @ t2_tr

    return r2_aaaa

def compute_mult_transpose_numpy(a19_aaaa,t2_aaaa):
   
    r2_aaaa  =  a19_aaaa @ np.einsum('ij->ji',t2_aaaa)
    return r2_aaaa

 
a19_aaaa = np.full((96,96), 2.0 ,dtype=float)
t2_aaaa = np.full((96,96), 2.0 ,dtype=float)


result = compute_mult_transpose(a19_aaaa,t2_aaaa)

exp_result = compute_mult_transpose_numpy(a19_aaaa,t2_aaaa)

np.testing.assert_array_equal(result,exp_result)
