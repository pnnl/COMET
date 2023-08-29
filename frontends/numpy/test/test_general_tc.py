
import numpy as np
import comet

#Testing general tensor contractions
@comet.compile(flags=None)
def ccsd_t1_general_tc(t1_a,a02):

    a03_a = comet.einsum('ai,m->aim', t1_a,a02)         #Gives backend errors with TTGT
    
    return a03_a


def expected_ccsd_t1_general_tc(t1_a,a02):

    a03_a = np.einsum('ai,m->aim', t1_a,a02)
    
    return a03_a


t1_a     = np.full((6,8), 0.1 ,dtype=float)
a02 = np.full((3), 1.3, dtype=float)

result = ccsd_t1_general_tc(t1_a,a02)
exp_result =  expected_ccsd_t1_general_tc(t1_a,a02)

np.testing.assert_equal(result, exp_result)
