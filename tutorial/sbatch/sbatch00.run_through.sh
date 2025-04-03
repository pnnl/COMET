#!/bin/bash
#SBATCH -A CENATE
#SBATCH -t 24:24:24
#SBATCH -N 1
#####SBATCH -p all
#####SBATCH -p dlv
#####SBATCH -p h100_shared
#####SBATCH -p h100
#####SBATCH -p a100
#SBATCH -p slurm
#SBATCH -J DSL_bench
#SBATCH -e out_sb.%x.%j.log
#SBATCH -o out_sb.%x.%j.log
#SBATCH --exclusive

#### sinfo -p <partition>
#### sinfo -N -r -l
#### srun -A CENATE -N 1 -t 20:20:20 --pty -u /bin/bash

#First make sure the module commands are available.
source /etc/profile.d/modules.sh

#Set up your environment you wish to run in with module commands.
module purge
# module load rocm/5.6.0
#module load cuda/12.3
# Modules needed by Orca
module load gcc/11.2.0 binutils/2.35 cmake/3.29.0
#module load openmpi/4.1.4
#module load mkl
module list &> _modules.lis_
cat _modules.lis_
/bin/rm -f _modules.lis_

#Python version
echo
echo "python version"
echo
command -v python
python --version

#Next unlimit system resources, and set any other environment variables you need.
ulimit -s unlimited
echo
echo limits
echo
ulimit -a

#Is extremely useful to record the modules you have loaded, your limit settings,
#your current environment variables and the dynamically load libraries that your executable
#is linked against in your job output file.
#echo
#echo "loaded modules"
#echo
#module list &> _modules.lis_
#cat _modules.lis_
#/bin/rm -f _modules.lis_
#echo
#echo limits
#echo
#ulimit -a
echo
echo "Environment Variables"
echo
printenv
# echo
# echo "ldd output"
# echo
# ldd your_executable

#Now you can put in your parallel launch command.
#For each different parallel executable you launch we recommend
#adding a corresponding ldd command to verify that the environment
#that is loaded corresponds to the environment the executable was built in.

start=$(date +%s.%N)

cd "/people/peng599/pppp/twotruth_project/COMET-indextree-rewrite-performance_mac/tutorial/session-1/cometdsl/run"
bash run_cometdsl.sh

end=$(date +%s.%N)
exe_time=$(echo "${end} - ${start}" | bc -l)
echo ""
echo "sbatch_exe_time(s): ${exe_time}"