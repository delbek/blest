#!/bin/bash
#SBATCH -J out
#SBATCH --account=proj13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=kolyoz-cuda
#SBATCH --gres=gpu:1
#SBATCH -C H100
#SBATCH --mem=1000G
#SBATCH --time=3-00:00:00
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

module load lib/cuda/13.0
if [ $? -ne 0 ]; then
  echo "Failed to load lib/cuda/13.0"
  exit 1
fi

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p ${repo_directory}build
cd ${repo_directory}build
cmake ..
make
cd ..

srun ./build/sutensor
#srun valgrind --tool=memcheck --leak-check=no --show-leak-kinds=none --track-origins=no --read-var-info=yes --num-callers=50 --error-limit=no ./build/sutensor
#srun ncu --config-file off --export /arf/home/delbek/profiler_reports/profile%i.ncu-rep --force-overwrite --section ComputeWorkloadAnalysis --section InstructionStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section PmSampling --section PmSampling_WarpStates --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_HierarchicalDoubleRooflineChart --section SpeedOfLight_HierarchicalHalfRooflineChart --section SpeedOfLight_HierarchicalSingleRooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart --section SpeedOfLight_RooflineChart --section WarpStateStats --section WorkloadDistribution --import-source yes --source-folder /arf/home/delbek/sutensor/ /arf/home/delbek/sutensor/build/sutensor
#srun compute-sanitizer --tool memcheck ./build/sutensor
