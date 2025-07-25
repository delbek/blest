#!/bin/bash
#SBATCH -J out
#SBATCH --account=proj13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=kolyoz-cuda
#SBATCH --gres=gpu:1
#SBATCH --time=0-0:20:00
#SBATCH --output=/arf/home/delbek/sutensor/res/out-%j.out
#SBATCH --error=/arf/home/delbek/sutensor/res/out-%j.err
#SBATCH --export=NONE

module purge
unset SLURM_EXPORT_ENV

source /etc/profile.d/modules.sh

repo_directory="/arf/home/delbek/sutensor/"

module use /arf/sw/modulefiles
module load comp/cmake/3.31.1
if [ $? -ne 0 ]; then
  echo "Failed to load comp/cmake/3.31.1"
  exit 1
fi

module load comp/gcc/12.3.0
if [ $? -ne 0 ]; then
  echo "Failed to load comp/gcc/12.3.0"
  exit 1
fi

module load lib/cuda/12.4
if [ $? -ne 0 ]; then
  echo "Failed to load lib/cuda/12.4"
  exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

mkdir -p ${repo_directory}build
cd ${repo_directory}build
cmake ..
make
cd ..

srun ./build/sutensor
#srun compute-sanitizer --tool memcheck ./build/sutensor
