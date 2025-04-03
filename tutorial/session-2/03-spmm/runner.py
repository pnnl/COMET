import subprocess
import sys
matrices = ['cant.mtx', 'pdb1HYS.mtx', 'rma10.mtx', 'shipsec1.mtx']
for m in matrices:
    p = subprocess.run('nvprof python3 ' + sys.argv[1] +' ' + m, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,close_fds=False)
    end = p.stderr.decode().find('API')
    print(p.stderr.decode()[:end])
    