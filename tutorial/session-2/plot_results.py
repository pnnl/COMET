import matplotlib.pyplot as plt
import sys

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
            print(app[:-3], size)
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
                    cublas_time += time_in_ms
        elif line.strip() == "" and not completed_size:
            sizes.append(int(size))
            cublas_times.append(cublas_time)
            cometpy_times.append(cometpy_time)
            completed_size = True
            print("found empty line")
            gpu_activies_found = False
        elif gpu_activies_found:
            segments = line.split()
            print(segments)
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
plt.figure()
plt.plot(sizes, cometpy_times, label='CometPy')
plt.plot(sizes, cublas_times, label='cuBLAS')
plt.title(app)
plt.xlabel('Size M=N=K')
plt.ylabel('Time(ms)')
plt.legend()
plt.tight_layout()
plt.savefig(app)


if 'matrix_mult' in app:

    cometpy_flops = [2*s*s*s*1e-9/(t*1e-3) for (s, t) in zip(sizes, cometpy_times)]
    cublas_flops = [2*s*s*s*1e-9/(t*1e-3) for (s, t) in zip(sizes, cublas_times)]
    plt.figure()
    plt.plot(sizes, cometpy_flops, label='CometPy')
    plt.plot(sizes, cublas_flops, label='cuBLAS')
    plt.title(app+'_throughput')
    plt.xlabel('Size M=N=K')
    plt.ylabel('GFLOPs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(app+'_throughput')