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


def run_numpy(L0,L1,L2):
	C =  ((L1 @ L2) * L0).sum()
	return C


#A = sp.sparse.csr_array(sp.io.mmread("../../../data/shipsec1.mtx"))
sparse_matrix_file_input1 = os.getenv('SPARSE_FILE_NAME0')
A = sp.sparse.csr_array(sp.io.mmread(sparse_matrix_file_input1))
L0 = sp.sparse.csr_array(sp.sparse.tril(A, format='csr'))

# Measure the execution time
for _ in range(num_runs):
	start_time = perf_counter()            # Start the timer	
	expected_result = run_numpy(L0,L0,L0)
	end_time = perf_counter()		       # End the timer

	total_time_numpy_opt += end_time - start_time  # Accumulate the execution time

average_time_numpy = total_time_numpy_opt / num_runs
print(f"Average Execution Time for Numpy: {average_time_numpy:.6f} seconds")


