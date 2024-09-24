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

max_size_small = 64
max_size_large = 128

############# NumPy ##########
def run_numpy_no_opt(v,t2):
	i0 = np.einsum('icmn,mnca->ia', v,t2)
	return i0

def run_numpy_opt(v,t2):
	i0 = np.einsum('icmn,mnca->ia', v,t2, optimize=True)
	return i0

############# COMET ##########
@comet.compile(flags=None)
def run_comet_no_opt(v,t2):
	i0 = comet.einsum('icmn,mnca->ia', v,t2)
	return i0

@comet.compile(flags="--convert-tc-to-ttgt") #opt1
def run_comet_opt1(v,t2):
	i0 = comet.einsum('icmn,mnca->ia', v,t2)
	return i0


#Inputs
v = np.full([max_size_small, max_size_small, max_size_large, max_size_large], 2.3,  dtype=float)
t2 = np.full([max_size_large, max_size_large, max_size_small, max_size_large], 3.4,  dtype=float)
i0 = np.full([max_size_small, max_size_large], 0.0,  dtype=float)

############################################################
############# NumPy and COMET NON OPTIMIZED CODE ##########
############################################################
#Measure the execution time without optimizations
for _ in range(num_runs):
	start_time = perf_counter()        # Start the timer	
	expected_result = run_numpy_no_opt(v,t2)  # Call the function
	end_time = perf_counter()		   # End the timer

	total_time_numpy_no_opt += end_time - start_time  # Accumulate the execution time

average_time_numpy = total_time_numpy_no_opt / num_runs
print(f"Average Execution Time for Numpy *WITHOUT* Optimizations: {average_time_numpy:.6f} seconds")

for _ in range(num_runs):
	start_time = perf_counter()        			# Start the timer
	result_with_jit = run_comet_no_opt(v,t2)
	end_time = perf_counter()		            # End the timer

	total_time_comet_no_opt += end_time - start_time  # Accumulate the execution time

average_time_comet = total_time_comet_no_opt / num_runs
print(f"Average Execution Time for COMET *WITHOUT* Optimizations: {average_time_comet:.6f} seconds\n")


############################################################
############# NumPy and COMET OPTIMIZED CODE ##########
############################################################
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
	result_with_jit = run_comet_opt1(v,t2)
	end_time = perf_counter()		            # End the timer

	total_time_comet_opt += end_time - start_time  # Accumulate the execution time

average_time_comet = total_time_comet_opt / num_runs
print(f"Average Execution Time for COMET Partial Optimization *LEVEL 1*: {average_time_comet:.6f} seconds")

