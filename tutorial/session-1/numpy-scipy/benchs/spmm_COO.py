import time
import numpy as np
import scipy as sp
from cometpy import comet
from time import perf_counter
from threadpoolctl import threadpool_info
import os

info = threadpool_info()
for lib in info:
    print(lib)

# Run the function X times and calculate the average time
num_runs = 10
total_time_numpy_opt = 0
total_time_comet_opt = 0

dense_matrix_col = 256

def run_numpy(A,B):
	C = A @ B 
	return C

@comet.compile(flags=None)
def run_comet(A,B):
	C = A @ B 
	return C


# A = sp.sparse.coo_array(sp.io.mmread("../../../data/shipsec1.mtx"))
sparse_matrix_file = os.getenv('SPARSE_FILE_NAME0')
A = sp.sparse.coo_array(sp.io.mmread(sparse_matrix_file))
B = np.full([A.shape[1], dense_matrix_col], 1.7,  dtype=float)
C = np.full([A.shape[0], dense_matrix_col], 0.0,  dtype=float)

# Measure the execution time
for _ in range(num_runs):
	start_time = perf_counter()            # Start the timer	
	expected_result = run_numpy(A,B)
	end_time = perf_counter()		       # End the timer

	total_time_numpy_opt += end_time - start_time  # Accumulate the execution time

average_time_numpy = total_time_numpy_opt / num_runs
print(f"Average Execution Time for Numpy: {average_time_numpy:.6f} seconds")


for _ in range(num_runs):
	start_time = perf_counter()        			# Start the timer
	result_comet = run_comet(A,B)
	end_time = perf_counter()		            # End the timer

	total_time_comet_opt += end_time - start_time  # Accumulate the execution time

average_time_comet = total_time_comet_opt / num_runs
print(f"Average Execution Time for COMET: {average_time_comet:.6f} seconds")


# if sp.sparse.issparse(expected_result):
# 	expected_result = expected_result.todense()
# 	result_with_jit = result_comet.todense()
# np.testing.assert_almost_equal(result_comet, expected_result,2)
