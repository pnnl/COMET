import time
import numpy as np
import scipy as sp
from cometpy import comet
from time import perf_counter

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def run_numpy(L0,L1,L2):
	C =  ((L1 @ L2) * L0)
	return C

@comet.compile(flags="--opt-comp-workspace")
def run_comet_with_jit(L0,L1,L2):
	C =  ((L1 @ L2) * L0)
	return C

A = sp.sparse.csr_array(sp.io.mmread("../data/shipsec1.mtx"))
L0 = sp.sparse.csr_array(sp.sparse.tril(A, format='csr'))


# Measure the execution time
start_time = perf_counter()
expected_result = run_numpy(L0,L0,L0)
end_time = perf_counter()
execution_time = end_time - start_time
print(f"Execution Time for Numpy: {execution_time} seconds")

# Measure the execution time
start_time = perf_counter()
result_with_jit = run_comet_with_jit(L0,L0,L0)
end_time = perf_counter()
execution_time = end_time - start_time
print(f"Execution Time for COMET: {execution_time} seconds")


# if sp.sparse.issparse(expected_result):
# 	expected_result = expected_result.todense()
# if sp.sparse.issparse(result_with_jit):
# 	result_with_jit = result_with_jit.todense()
# np.testing.assert_almost_equal(result_with_jit, expected_result)