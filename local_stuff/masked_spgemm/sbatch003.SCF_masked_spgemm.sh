#!/bin/bash
#SBATCH -A CENATE
#SBATCH -t 9:0:0
#SBATCH -N 1
#SBATCH -J masked_spgemm

#First make sure the module commands are available.
source /etc/profile.d/modules.sh

#Set up your environment you wish to run in with module commands.
module purge
module load gcc/8.1.0 python/3.7.0

#Next unlimit system resources, and set any other environment variables you need.
ulimit

#Is extremely useful to record the modules you have loaded, your limit settings,
#your current environment variables and the dynamically load libraries that your executable
#is linked against in your job output file.
echo
echo "loaded modules"
echo
module list &> _modules.lis_
cat _modules.lis_
/bin/rm -f _modules.lis_
echo
echo limits
echo
ulimit -a
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
project_path="/people/peng599/pppp/clion/COMET_masking_mac/local_stuff/masked_spgemm"
command="bash test000.SCF_masked_spgemm.sh"

cd "${project_path}" || exit
eval "${command}"