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
num_runs = 10
total_time_numpy_no_opt = 0
total_time_comet_no_opt = 0

total_time_numpy_opt = 0
total_time_comet_opt = 0

############# NumPy ##########
def run_numpy_no_opt(v,t2):
	i0 = np.einsum('dca,bd->abc', v,t2)
	return i0

def run_numpy_opt(v,t2):
	i0 = np.einsum('dca,bd->abc', v,t2, optimize=True)
	return i0

############# COMET ##########
@comet.compile(flags=None)
def run_comet_no_opt(v,t2):
	i0 = comet.einsum('dca,bd->abc', v,t2)
	return i0

# @comet.compile(flags="--convert-tc-to-ttgt") #opt1
# @comet.compile(flags="--opt-matmul-tiling --convert-tc-to-ttgt") #opt2
@comet.compile(flags="--opt-bestperm-ttgt --opt-matmul-tiling  --opt-matmul-mkernel  --opt-dense-transpose --convert-tc-to-ttgt") #opt3
def run_comet_opt(v,t2):
	i0 = comet.einsum('dca,bd->abc', v,t2)
	return i0

#Inputs
v = np.full([312, 296, 312], 2.3,  dtype=float)
t2 = np.full([312, 312], 3.4,  dtype=float)
i0 = np.full([312, 312, 296], 0.0,  dtype=float)


# Measure the execution time without optimizations
# for _ in range(num_runs):
# 	start_time = perf_counter()        # Start the timer	
# 	expected_result = run_numpy_no_opt(v,t2)  # Call the function
# 	end_time = perf_counter()		   # End the timer

# 	total_time_numpy_no_opt += end_time - start_time  # Accumulate the execution time

# average_time_numpy = total_time_numpy_no_opt / num_runs
# print(f"Average Execution Time for Numpy *WITHOUT* Optimizations: {average_time_numpy:.6f} seconds")

# for _ in range(num_runs):
# 	start_time = perf_counter()        			# Start the timer
# 	result_with_jit = run_comet_no_opt(v,t2)
# 	end_time = perf_counter()		            # End the timer

# 	total_time_comet_no_opt += end_time - start_time  # Accumulate the execution time

# average_time_comet = total_time_comet_no_opt / num_runs
# print(f"Average Execution Time for COMET *WITHOUT* Optimizations: {average_time_comet:.6f} seconds")

# Measure the execution time with optimizations enabled
for _ in range(num_runs):
	start_time = perf_counter()            # Start the timer	
	expected_result = run_numpy_opt(v,t2)  # Call the function
	end_time = perf_counter()		       # End the timer

	total_time_numpy_opt += end_time - start_time  # Accumulate the execution time

average_time_numpy = total_time_numpy_opt / num_runs
print(f"Average Execution Time for Numpy *WITH* Optimizations: {average_time_numpy:.6f} seconds")

for _ in range(num_runs):
	start_time = perf_counter()        			# Start the timer
	result_with_jit = run_comet_opt(v,t2)
	end_time = perf_counter()		            # End the timer

	total_time_comet_opt += end_time - start_time  # Accumulate the execution time

average_time_comet = total_time_comet_opt / num_runs
print(f"Average Execution Time for COMET *WITH* Optimizations: {average_time_comet:.6f} seconds")


# if sp.sparse.issparse(expected_result):
# 	expected_result = expected_result.todense()
# 	result_with_jit = result_with_jit.todense()
# np.testing.assert_almost_equal(result_with_jit, expected_result)
