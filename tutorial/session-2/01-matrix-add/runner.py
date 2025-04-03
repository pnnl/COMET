import subprocess
import sys

for i in range(128, 4096+1, 128):
    p = subprocess.run('nvprof python3 ' + sys.argv[1] +' ' + str(i), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,close_fds=False)
    end = p.stderr.decode().find('API')
    print(p.stderr.decode()[:end])
    