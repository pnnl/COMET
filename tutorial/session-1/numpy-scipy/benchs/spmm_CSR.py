import time
import numpy as np
import scipy as sp
from cometpy import comet
from time import perf_counter
from threadpoolctl import threadpool_info

info = threadpool_info()
for lib in info:
    print(lib)

# Run the function 10 times and calculate the average time
num_runs = 10
total_time_numpy_opt = 0
total_time_comet_opt = 0

def run_numpy(A,B):
	C = A @ B 
	return C

@comet.compile(flags=None)
def run_comet_with_jit(A,B):
	C = A @ B 
	return C

B = sp.sparse.csr_array(sp.io.mmread("../../../data/shipsec1.mtx"))
A = np.full([4, B.shape[0]], 1.7,  dtype=float)
C = np.full([4, B.shape[1]], 0.0,  dtype=float)

# Measure the execution time
start_time = perf_counter()            # Start the timer	
expected_result = run_numpy(A,B)
end_time = perf_counter()		       # End the timer
print(f"Time for Numpy: {end_time-start_time:.6f} seconds")

# Measure the execution time
start_time = perf_counter()            # Start the timer	
result_with_jit = run_comet_with_jit(A,B)
end_time = perf_counter()		       # End the timer
print(f"Time for COMET: {end_time-start_time:.6f} seconds")

if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result,2)
