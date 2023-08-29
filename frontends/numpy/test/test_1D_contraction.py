
import numpy as np
import comet

#Testing Tensor Contractions with 1D input/output
@comet.compile(flags=None)
def compute_einsum_1D_comet(A,B):
    
    C = comet.einsum('ai,iam->m',A,B)
    return C


def compute_einsum_1D_numpy(A,B):
    
    C_expected =  np.einsum('ai,iam->m',A,B)    
    return C_expected


A = np.ones([4,5], dtype=float)
B = np.full((5,4,2), 3, dtype=float)
 
t2_result = compute_einsum_1D_comet(A,B)
t2_expected_result = compute_einsum_1D_numpy(A,B)

np.testing.assert_array_equal(t2_result, t2_expected_result)


@comet.compile(flags=None)
def compute_einsum_augassign(t1_a,t1_b,ch_a_ov,ch_b_ov):
    
    a02    =  comet.einsum('ai,iam->m',t1_a,ch_a_ov)
    a02   +=  comet.einsum('ai,iam->m',t1_b,ch_b_ov)

    return a02

def compute_einsum_augassign_numpy(t1_a,t1_b,ch_a_ov,ch_b_ov):
    
    a02    =  np.einsum('ai,iam->m',t1_a,ch_a_ov)
    a02   +=  np.einsum('ai,iam->m',t1_b,ch_b_ov)

    return a02


t1_a     = np.full((6,8), 1.0 ,dtype=float)
t1_b     = np.full((8,6),2.0 ,dtype=float)
ch_a_ov = np.full((8,6,3),1.2 ,dtype=float)             
ch_b_ov = np.full((6,8,3),1.3 ,dtype=float) 

augassign_result = compute_einsum_augassign(t1_a,t1_b,ch_a_ov,ch_b_ov)
augassign_expected_result = compute_einsum_augassign_numpy(t1_a,t1_b,ch_a_ov,ch_b_ov)

#With Comet the result if off by 8.52651283e-14, hence using "almost equal". Floating point accuracy might be the reason.
np.testing.assert_almost_equal(augassign_result, augassign_expected_result)