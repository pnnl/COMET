
import numpy as np
import comet
import scipy as scp
#Testing elementwise multiplication

@comet.compile(flags=None)
def compute_tensor_elwsmult(a19_aaaa,t2_aaaa):

    r2_aaaa  =  a19_aaaa * t2_aaaa
   
    # r2_aaaa  =  comet.multiply(a19_aaaa,t2_aaaa)

    return r2_aaaa

def compute_tensor_elwsmult_numpy(a19_aaaa,t2_aaaa):
   
    r2_aaaa  =  np.multiply(a19_aaaa,t2_aaaa)

    return r2_aaaa

 
a19_aaaa = np.full((96,96), 2.0 ,dtype=float)
t2_aaaa = np.full((96,96), 2.0 ,dtype=float)

#Returns COO
result = compute_tensor_elwsmult(scp.sparse.csr_matrix(a19_aaaa),t2_aaaa)

exp_result = compute_tensor_elwsmult_numpy(a19_aaaa,t2_aaaa)

np.testing.assert_array_equal(result,exp_result)
