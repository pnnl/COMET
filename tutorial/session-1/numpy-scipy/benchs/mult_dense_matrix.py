import time
import numpy as np
import scipy as sp
from cometpy import comet
from time import perf_counter
from threadpoolctl import threadpool_info

# Run this experiment with export OMP_NUM_THREADS=1
# print(np.__config__.show())
info = threadpool_info()
for lib in info:
    print(lib)
	
# Run the function 10 times and calculate the average time
num_runs = 3
max_size = 4096
total_time_numpy_no_opt = 0
total_time_comet_no_opt = 0

total_time_numpy_opt = 0
total_time_comet_opt = 0

############# NumPy ##########
def run_numpy_no_opt(A,B):
	C = np.einsum('ij,jk->ik', A,B)
	return C

def run_numpy_opt(A,B):
	C = np.einsum('ij,jk->ik', A,B, optimize=True)
	return C

############# COMET ##########
@comet.compile(flags=None)
def run_comet_no_opt(A,B):
	C = comet.einsum('ij,jk->ik', A,B)
	return C

@comet.compile(flags="--convert-tc-to-ttgt -opt-matmul-tiling -opt-matmul-mkernel ")
def run_comet_opt(A,B):
	C = comet.einsum('ij,jk->ik', A,B)

	return C


A = np.full([max_size, max_size], 2.2,  dtype=float)
B = np.full([max_size, max_size], 3.4,  dtype=float)
C = np.full([max_size, max_size], 0.0,  dtype=float)

# Measure the execution time for non optimized code
for _ in range(num_runs):
	start_time = perf_counter()               # Start the timer
	expected_result = run_numpy_no_opt(A,B)   # Call the function
	end_time = perf_counter()		          # End the timer

	total_time_numpy_no_opt += end_time - start_time  # Accumulate the execution time

average_time_numpy = total_time_numpy_no_opt / num_runs
print(f"Average Execution Time for Numpy WITHOUT Optimization: {average_time_numpy:.6f} seconds")

for _ in range(num_runs):
	start_time = perf_counter()        			# Start the timer
	result_with_jit = run_comet_no_opt(A,B)     # Call the function
	end_time = perf_counter()		   			# End the timer

	total_time_comet_no_opt += end_time - start_time  # Accumulate the execution time

average_time_comet = total_time_comet_no_opt / num_runs
print(f"Average Execution Time for COMET WITHOUT Optimization: {average_time_comet:.6f} seconds")


# Measure the execution time for optimized code
for _ in range(num_runs):
	start_time = perf_counter()               # Start the timer
	expected_result = run_numpy_opt(A,B)      # Call the function
	end_time = perf_counter()		          # End the timer

	total_time_numpy_opt += end_time - start_time  # Accumulate the execution time

average_time_numpy = total_time_numpy_opt / num_runs
print(f"Average Execution Time for Numpy WITH Optimization: {average_time_numpy:.6f} seconds")

for _ in range(num_runs):
	start_time = perf_counter()        			# Start the timer
	result_with_jit = run_comet_opt(A,B)        # Call the function
	end_time = perf_counter()		   			# End the timer

	total_time_comet_opt += end_time - start_time  # Accumulate the execution time

average_time_comet = total_time_comet_opt / num_runs
print(f"Average Execution Time for COMET WITH Optimization: {average_time_comet:.6f} seconds")

#Validation
# if sp.sparse.issparse(expected_result):
# 	expected_result = expected_result.todense()
# 	result_with_jit = result_with_jit.todense()
# np.testing.assert_almost_equal(result_with_jit, expected_result)
