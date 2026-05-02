#!/bin/bash
#SBATCH --job-name=icon_etopo_global
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/icon_etopo_%j.log
#SBATCH --error=logs/icon_etopo_%j.err

# SLURM submission script for ICON ETOPO global processing
# Optimized for: 128 cores, 256 GB RAM single node

echo "========================================="
echo "ICON ETOPO Global Processing"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Memory: 256 GB"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust for your HPC system)
# module load anaconda3  # or your Python environment
# module load netcdf4

# Activate conda environment
# source activate playground  # or your environment name

# Set OpenMP threads to 1 (we use Dask for parallelism)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Increase file descriptor limits (NetCDF files)
ulimit -n 4096

# Run the HPC-optimized script
echo "Starting ICON ETOPO processing..."
python3 -m runs.icon_etopo_global

exit_code=$?

echo ""
echo "========================================="
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "========================================="

exit $exit_code
