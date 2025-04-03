import matplotlib.pyplot as plt
import sys

import numpy as np

sizes = []
cometpy_times = []
cublas_times = []
app = None
completed_size = False
gpu_activies_found = False
with open(sys.argv[1], 'r') as f:
    for line in f:
        if 'Profiling application' in line:
            app, size = line.split()[4:6]
            app  = app[:-3]
            print(app, size)
            cometpy_time = 0.0
            cublas_time = 0.0
            completed_size = False
        elif 'GPU activities:' in line:
            gpu_activies_found = True
            segments = line.split()
            kernel  = segments[8]
            avg = segments[5]
            if '[CUDA' in kernel :
                pass
            else:
                time_in_ms = 0.0
                if 'us' in avg :
                    time_in_ms = float(avg[:-2])/1000
                elif 'ms' in avg :
                    time_in_ms = float(avg[:-2])
                if 'cometpy' in kernel:
                    cometpy_time += time_in_ms
                else:
                    print("KERNEL: ", kernel)
                    cublas_time += time_in_ms
        elif line.strip() == "" and not completed_size:
            sizes.append(size)
            print(cublas_time)
            cublas_times.append(cublas_time)
            cometpy_times.append(cometpy_time)
            completed_size = True
            print("found empty line")
            gpu_activies_found = False
        elif gpu_activies_found:
            segments = line.split()
            # print(segments)
            kernel  = segments[6]
            avg = segments[3]
            if '[CUDA' in kernel or '[CUDA' in kernel :
                pass
            else:
                time_in_ms = 0.0
                if 'us' in avg :
                    time_in_ms = float(avg[:-2])/1000
                elif 'ms' in avg :
                    time_in_ms = float(avg[:-2])
                if 'cometpy' in kernel:
                    cometpy_time += time_in_ms
                else:
                    cublas_time += time_in_ms

print(app)
print(sizes)
print(cometpy_times)
print(cublas_times)

# Set the positions of the bars on the x-axis
x = np.arange(len(sizes))  # Create an array for the number of categories
width = 0.35  # Width of the bars

# Create the bar graph
plt.bar(x - width/2, cometpy_times, width, label='CometPy')
plt.bar(x + width/2, cublas_times, width, label='cuBLAS')

# Add title and labels
plt.xlabel('Matrix')
plt.ylabel('Time(ms)')
plt.xticks(x, sizes)  # Replace x-axis with category labels

# Add a legend
plt.legend()

# Display the graph
# plt.show()
# plt.figure()
# plt.bar(sizes, cometpy_times, label='CometPy')
# plt.bar(sizes, cublas_times, label='cuBLAS')
# plt.title(app)
# plt.xlabel('Size M=N=K')
# plt.ylabel('Time(ms)')
# plt.legend()
# plt.tight_layout()
plt.savefig(app)